"""
init_db.py
────────────────────────────────────────────────────────────────────────────
DB 초기화 및 데이터 적재 스크립트

실행:
    python init_db.py [--db chroma|pinecone|both] [--reset]

주요 동작:
  1. 데이터셋 로드 (250개 매물)
  2. 이미 DB에 존재하는 ID 확인 (중복 체크 → 불필요한 재임베딩 방지)
  3. 신규 항목만 배치 임베딩 후 DB에 삽입
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from data.properties import PROPERTIES
from db.chroma_client import get_chroma_client, get_collection, BATCH_SIZE
from db.pinecone_client import get_pinecone_index, embed_texts, get_openai_client


# ── ChromaDB 초기화 ────────────────────────────────────────────────────────

def init_chroma(reset: bool = False):
    """
    ChromaDB에 데이터 적재.
    - PersistentClient 사용: 디스크에 영구 저장
    - 기존에 존재하는 ID는 건너뜀 (중복 체크)
    - 신규 항목만 배치로 삽입 (OpenAI 임베딩 비용 최소화)
    """
    print("\n========== ChromaDB 초기화 시작 ==========")
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection("real_estate")
            print("[ChromaDB] 기존 컬렉션 삭제 완료.")
        except Exception:
            pass

    collection = get_collection(client)
    print(f"[ChromaDB] 컬렉션 '{collection.name}' 준비 완료.")

    # ── 기존 항목 확인 (ID 기반 중복 체크) ───────────────────────────────
    all_ids  = [p["id"] for p in PROPERTIES]
    existing = set()

    try:
        result   = collection.get(ids=all_ids, include=[])
        existing = set(result["ids"])
    except Exception:
        pass

    new_properties = [p for p in PROPERTIES if p["id"] not in existing]
    print(f"[ChromaDB] 전체: {len(PROPERTIES)}개 | "
          f"기존: {len(existing)}개 | "
          f"신규 삽입: {len(new_properties)}개")

    if not new_properties:
        print("[ChromaDB] 모든 데이터가 이미 존재합니다. 적재를 건너뜁니다.")
        return collection

    # ── 배치 삽입 ─────────────────────────────────────────────────────────
    # ChromaDB는 text를 주면 등록된 embedding_function이 자동으로 벡터화함
    for i in range(0, len(new_properties), BATCH_SIZE):
        batch = new_properties[i : i + BATCH_SIZE]

        ids       = [p["id"] for p in batch]
        documents = [p["title"] + ". " + p["description"] for p in batch]  # 임베딩 대상 텍스트
        metadatas = [
            {
                "property_type": p["property_type"],
                "district":      p["district"],
                "price_eok":     p["price_eok"],
                "area_m2":       p["area_m2"],
                "rooms":         p["rooms"],
                "year_built":    p["year_built"],
                "floor":         p["floor"],
                "has_parking":   str(p["has_parking"]),   # ChromaDB 메타데이터: bool → str
                "title":         p["title"],
            }
            for p in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"  [ChromaDB] 배치 {i // BATCH_SIZE + 1}: "
              f"{i + 1}~{min(i + BATCH_SIZE, len(new_properties))}번 항목 삽입 완료")

    print(f"[ChromaDB] 적재 완료. 총 항목 수: {collection.count()}")
    return collection


# ── Pinecone 초기화 ────────────────────────────────────────────────────────

def init_pinecone(reset: bool = False):
    """
    Pinecone에 데이터 적재.
    - 외부 임베딩 주입 방식: OpenAI로 벡터를 직접 생성 후 Pinecone에 upsert
    - upsert의 멱등성: 동일 ID 데이터를 여러 번 upsert해도 결과가 동일
    - 기존에 존재하는 ID는 건너뜀 (불필요한 임베딩 API 비용 절감)
    """
    print("\n========== Pinecone 초기화 시작 ==========")
    index = get_pinecone_index()

    if reset:
        print("[Pinecone] 인덱스 초기화(delete all)...")
        index.delete(delete_all=True)
        import time; time.sleep(3)

    # ── 기존 항목 확인 ────────────────────────────────────────────────────
    stats    = index.describe_index_stats()
    existing_count = stats.get("total_vector_count", 0)
    print(f"[Pinecone] 현재 저장된 벡터 수: {existing_count}")

    all_ids  = [p["id"] for p in PROPERTIES]

    # Pinecone fetch로 기존 ID 확인 (배치 100개 단위)
    existing_ids = set()
    for i in range(0, len(all_ids), 100):
        batch_ids = all_ids[i : i + 100]
        result    = index.fetch(ids=batch_ids)
        existing_ids.update(result.vectors.keys())

    new_properties = [p for p in PROPERTIES if p["id"] not in existing_ids]
    print(f"[Pinecone] 전체: {len(PROPERTIES)}개 | "
          f"기존: {len(existing_ids)}개 | "
          f"신규 삽입: {len(new_properties)}개")

    if not new_properties:
        print("[Pinecone] 모든 데이터가 이미 존재합니다. 적재를 건너뜁니다.")
        return index

    # ── 배치 임베딩 및 upsert ─────────────────────────────────────────────
    openai_client = get_openai_client()

    for i in range(0, len(new_properties), BATCH_SIZE):
        batch = new_properties[i : i + BATCH_SIZE]
        texts = [p["title"] + ". " + p["description"] for p in batch]

        # 1. OpenAI로 벡터 생성 (Pinecone 외부 주입 방식)
        vectors = embed_texts(texts, openai_client)

        # 2. Pinecone upsert 형식으로 변환
        upsert_data = []
        for prop, vector in zip(batch, vectors):
            upsert_data.append({
                "id":     prop["id"],
                "values": vector,
                "metadata": {
                    "property_type": prop["property_type"],
                    "district":      prop["district"],
                    "price_eok":     prop["price_eok"],
                    "area_m2":       prop["area_m2"],
                    "rooms":         prop["rooms"],
                    "year_built":    prop["year_built"],
                    "floor":         prop["floor"],
                    "has_parking":   prop["has_parking"],
                    "title":         prop["title"],
                    "description":   prop["description"],
                }
            })

        # 3. upsert: 동일 ID가 존재하면 덮어씀 (멱등성)
        index.upsert(vectors=upsert_data)
        print(f"  [Pinecone] 배치 {i // BATCH_SIZE + 1}: "
              f"{i + 1}~{min(i + BATCH_SIZE, len(new_properties))}번 항목 upsert 완료")

    import time; time.sleep(2)  # Pinecone 인덱싱 반영 대기
    final_stats = index.describe_index_stats()
    print(f"[Pinecone] 적재 완료. 총 벡터 수: {final_stats['total_vector_count']}")
    return index


# ── 멀티 컬렉션 초기화 (확장 기능: 다중 컬렉션 / 도메인 분리) ──────────────

def init_chroma_multi_collection(client=None):
    """
    ChromaDB 다중 컬렉션 구현 (확장 기능):
    property_type별로 별도 컬렉션에 저장 → 멀티테넌시/도메인 분리 시연.
    """
    from db.chroma_client import get_collection

    if client is None:
        client = get_chroma_client()

    type_map = {}
    for p in PROPERTIES:
        pt = p["property_type"]
        type_map.setdefault(pt, []).append(p)

    collections = {}
    for prop_type, props in type_map.items():
        col_name   = f"real_estate_{prop_type}"
        collection = get_collection(client, col_name)
        print(f"[ChromaDB 다중컬렉션] '{col_name}' 준비 (현재: {collection.count()}개)")

        existing_result = collection.get(ids=[p["id"] for p in props], include=[])
        existing_ids    = set(existing_result["ids"])
        new_props       = [p for p in props if p["id"] not in existing_ids]

        if new_props:
            for i in range(0, len(new_props), BATCH_SIZE):
                batch = new_props[i : i + BATCH_SIZE]
                collection.add(
                    ids=[p["id"] for p in batch],
                    documents=[p["title"] + ". " + p["description"] for p in batch],
                    metadatas=[
                        {k: str(v) if isinstance(v, bool) else v
                         for k, v in {
                            "property_type": p["property_type"],
                            "district":      p["district"],
                            "price_eok":     p["price_eok"],
                            "area_m2":       p["area_m2"],
                            "rooms":         p["rooms"],
                            "year_built":    p["year_built"],
                            "floor":         p["floor"],
                            "has_parking":   p["has_parking"],
                            "title":         p["title"],
                         }.items()}
                        for p in batch
                    ]
                )
            print(f"  → {len(new_props)}개 신규 삽입")
        collections[prop_type] = collection

    return collections


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="부동산 벡터 DB 초기화")
    parser.add_argument("--db",    choices=["chroma", "pinecone", "both"],
                        default="both", help="초기화할 DB")
    parser.add_argument("--reset", action="store_true",
                        help="기존 데이터 삭제 후 재삽입")
    parser.add_argument("--multi", action="store_true",
                        help="ChromaDB 다중 컬렉션 초기화 (확장 기능)")
    args = parser.parse_args()

    if args.db in ("chroma", "both"):
        init_chroma(reset=args.reset)
        if args.multi:
            init_chroma_multi_collection()

    if args.db in ("pinecone", "both"):
        init_pinecone(reset=args.reset)

    print("\n✅ 초기화 완료!")
