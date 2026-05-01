"""
db/crud.py
────────────────────────────────────────────────────────────────────────────
ChromaDB & Pinecone CRUD 작업 구현

[핵심 개념] update vs upsert
───────────────────────────
  collection.update():
    - 지정 ID가 DB에 반드시 존재해야 함
    - 없으면 예외(InvalidCollectionException) 발생
    - 의도치 않은 신규 생성을 방지하고 싶을 때 사용
    - 예: 관리자가 특정 매물 가격을 수정할 때

  collection.upsert():
    - 지정 ID가 없으면 새로 생성, 있으면 덮어씀
    - 멱등성(Idempotency) 보장:
        동일한 upsert를 여러 번 실행해도 최종 결과는 항상 동일
    - 예: 외부 시스템에서 주기적으로 데이터를 동기화할 때

[Pinecone은 upsert로 통합]
  Pinecone은 별도 update API 없음. upsert가 create/update 모두 담당.
  동일 ID로 upsert 반복 시 마지막 값으로 덮어씌워지며 DB 상태 동일 유지.
────────────────────────────────────────────────────────────────────────────
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
# ChromaDB CRUD
# ══════════════════════════════════════════════════════════════════════════

def _to_chroma_meta(p: dict) -> dict:
    """매물 dict → ChromaDB 메타데이터 dict 변환 (bool → str)."""
    return {
        "property_type": p["property_type"],
        "district":      p["district"],
        "price_eok":     float(p["price_eok"]),
        "area_m2":       float(p["area_m2"]),
        "rooms":         int(p["rooms"]),
        "year_built":    int(p["year_built"]),
        "floor":         int(p["floor"]),
        "has_parking":   str(p["has_parking"]),   # ChromaDB: bool → str
        "title":         p["title"],
    }


def batch_insert_chroma(collection, properties: list[dict]):
    """
    [CREATE] 다수의 매물을 ChromaDB에 일괄 삽입.
    collection.add(): 이미 존재하는 ID 삽입 시 예외 발생.
    """
    ids       = [p["id"] for p in properties]
    documents = [p["title"] + ". " + p["description"] for p in properties]
    metadatas = [_to_chroma_meta(p) for p in properties]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[ChromaDB CREATE] {len(properties)}개 항목 삽입 완료.")


def get_by_id_chroma(collection, ids: list[str]) -> dict:
    """[READ - by ID] 특정 ID로 항목 조회."""
    return collection.get(ids=ids, include=["documents", "metadatas"])


def query_chroma(collection, query_text: str, n_results: int = 5,
                 where: Optional[dict] = None) -> dict:
    """
    [READ - Query] 시맨틱 유사도 기반 검색.
    where 파라미터가 있으면 하이브리드 검색, 없으면 순수 시맨틱 검색.
    """
    kwargs = dict(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def update_chroma(collection, prop_id: str,
                  new_metadata: dict = None, new_document: str = None):
    """
    [UPDATE] 기존 항목 수정.

    ★ update() 동작:
      - 지정 ID가 반드시 DB에 존재해야 함
      - 없으면 예외 발생 → 의도치 않은 신규 생성 방지
      - 안전하게 '기존 항목만' 수정하고 싶을 때 사용

    ※ upsert()와의 차이:
      - update(): 존재 확인 후 수정 (엄격)
      - upsert(): 없으면 생성, 있으면 덮어씀 (유연, 멱등)
    """
    # 존재 여부 사전 확인 (명시적 오류 메시지 제공)
    existing = collection.get(ids=[prop_id], include=[])
    if not existing["ids"]:
        raise ValueError(
            f"[ChromaDB UPDATE 실패] ID '{prop_id}' 가 존재하지 않습니다. "
            f"update()는 반드시 존재하는 항목만 수정 가능합니다. "
            f"신규 생성이 필요하면 upsert()를 사용하세요."
        )

    kwargs = {"ids": [prop_id]}
    if new_metadata:
        kwargs["metadatas"] = [new_metadata]
    if new_document:
        kwargs["documents"] = [new_document]

    collection.update(**kwargs)
    print(f"[ChromaDB UPDATE] ID '{prop_id}' 수정 완료.")


def upsert_chroma(collection, prop: dict):
    """
    [UPSERT] 멱등성 업서트.

    ★ upsert() 동작:
      - ID가 없으면 새로 생성
      - ID가 있으면 덮어씀
      - 동일한 데이터로 여러 번 호출해도 결과 항상 동일 = 멱등성(Idempotency)

    ※ update()와의 차이:
      - upsert(): 존재 여부 상관없이 항상 성공 (유연, 멱등)
      - update(): 없으면 예외 발생 (엄격, 안전)
    """
    collection.upsert(
        ids=[prop["id"]],
        documents=[prop["title"] + ". " + prop.get("description", "")],
        metadatas=[_to_chroma_meta(prop)]
    )
    print(f"[ChromaDB UPSERT] ID '{prop['id']}' upsert 완료 (멱등성 보장).")


def delete_chroma(collection, ids: list[str] = None, where: dict = None):
    """[DELETE] ID 또는 조건(where)으로 항목 삭제."""
    if ids:
        collection.delete(ids=ids)
        print(f"[ChromaDB DELETE] ID {ids} 삭제 완료.")
    elif where:
        collection.delete(where=where)
        print(f"[ChromaDB DELETE] 조건 {where} 해당 항목 삭제 완료.")
    else:
        raise ValueError("ids 또는 where 중 하나를 반드시 지정하세요.")


# ══════════════════════════════════════════════════════════════════════════
# Pinecone CRUD
# ══════════════════════════════════════════════════════════════════════════

def _get_embedding(text: str) -> list[float]:
    """단일 텍스트 임베딩 (Pinecone 단건 upsert / 쿼리용)."""
    client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def upsert_pinecone(index, prop: dict):
    """
    [CREATE / UPDATE / UPSERT] Pinecone upsert.

    ★ Pinecone upsert 동작:
      - Pinecone은 별도 update API가 없고 upsert로 통합
      - 동일 ID로 upsert 반복 시 마지막 값으로 덮어씌워짐
      - 멱등성 보장: 동일 데이터 반복 upsert → DB 상태 항상 동일
    """
    text   = prop["title"] + ". " + prop.get("description", "")
    vector = _get_embedding(text)

    index.upsert(vectors=[{
        "id":     prop["id"],
        "values": vector,
        "metadata": {
            "property_type": prop["property_type"],
            "district":      prop["district"],
            "price_eok":     float(prop["price_eok"]),
            "area_m2":       float(prop["area_m2"]),
            "rooms":         int(prop["rooms"]),
            "year_built":    int(prop["year_built"]),
            "floor":         int(prop["floor"]),
            "has_parking":   bool(prop["has_parking"]),
            "title":         prop["title"],
            "description":   prop.get("description", ""),
        }
    }])
    print(f"[Pinecone UPSERT] ID '{prop['id']}' upsert 완료.")


def get_by_id_pinecone(index, ids: list[str]) -> dict:
    """[READ - by ID] Pinecone fetch로 특정 ID 항목 조회."""
    return index.fetch(ids=ids)


def query_pinecone(index, query_text: str, n_results: int = 5,
                   filter_dict: Optional[dict] = None) -> dict:
    """
    [READ - Query] 시맨틱 유사도 기반 검색.
    filter_dict 있으면 하이브리드 검색, 없으면 순수 시맨틱 검색.
    """
    query_vector = _get_embedding(query_text)
    kwargs = dict(vector=query_vector, top_k=n_results, include_metadata=True)
    if filter_dict:
        kwargs["filter"] = filter_dict
    return index.query(**kwargs)


def delete_pinecone(index, ids: list[str] = None, filter_dict: dict = None):
    """[DELETE] ID 또는 조건(filter)으로 항목 삭제."""
    if ids:
        index.delete(ids=ids)
        print(f"[Pinecone DELETE] ID {ids} 삭제 완료.")
    elif filter_dict:
        index.delete(filter=filter_dict)
        print(f"[Pinecone DELETE] 조건 {filter_dict} 해당 항목 삭제 완료.")
    else:
        raise ValueError("ids 또는 filter_dict 중 하나를 반드시 지정하세요.")
