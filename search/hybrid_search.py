"""
search/hybrid_search.py
────────────────────────────────────────────────────────────────────────────
하이브리드 검색 구현 (시맨틱 + 메타데이터 필터링)

[시나리오 구성]
  ChromaDB 5개 시나리오 + Pinecone 3개 시나리오

[사용 연산자 — 6종 모두 포함]
  $eq   : 값이 정확히 일치 (시나리오 1, 3, 4, 5)
  $ne   : 값이 일치하지 않음 (시나리오 2)
  $gte  : 값 이상 (시나리오 1, 4, 5)
  $lte  : 값 이하 (시나리오 2, 3, 4)
  $and  : 복수 조건 AND 결합 (시나리오 1, 2, 4, 5)
  $or   : 복수 조건 OR 결합 (시나리오 3)
────────────────────────────────────────────────────────────────────────────
"""

from db.chroma_client import get_collection
from db.pinecone_client import get_pinecone_index
from db.crud import query_chroma, query_pinecone


# ══════════════════════════════════════════════════════════════════════════
# ChromaDB 하이브리드 검색 시나리오 (5개)
# ══════════════════════════════════════════════════════════════════════════

def scenario_1_chroma(n=5):
    """
    시나리오 1: 강남구 아파트 15억 이상
    연산자: $and( $eq, $eq, $gte )
    """
    q = "강남 고급 아파트 역세권 직주근접"
    col = get_collection()
    sem = query_chroma(col, q, n_results=n)
    hyb = query_chroma(col, q, n_results=n, where={
        "$and": [
            {"property_type": {"$eq": "아파트"}},    # $eq
            {"district":      {"$eq": "강남구"}},    # $eq
            {"price_eok":     {"$gte": 15.0}}        # $gte
        ]
    })
    return sem, hyb


def scenario_2_chroma(n=5):
    """
    시나리오 2: 15층 이상, 20억 이하, 상가 제외
    연산자: $and( $gte, $lte, $ne )
    """
    q = "한강 조망 고층 아파트 강변 산책"
    col = get_collection()
    sem = query_chroma(col, q, n_results=n)
    hyb = query_chroma(col, q, n_results=n, where={
        "$and": [
            {"floor":         {"$gte": 15}},         # $gte
            {"price_eok":     {"$lte": 20.0}},       # $lte
            {"property_type": {"$ne": "상가"}}        # $ne
        ]
    })
    return sem, hyb


def scenario_3_chroma(n=5):
    """
    시나리오 3: 5억 이하 OR 오피스텔
    연산자: $or( $lte, $eq )
    """
    q = "1인 가구 소형 저렴한 오피스텔 역세권"
    col = get_collection()
    sem = query_chroma(col, q, n_results=n)
    hyb = query_chroma(col, q, n_results=n, where={
        "$or": [
            {"price_eok":     {"$lte": 5.0}},        # $lte
            {"property_type": {"$eq": "오피스텔"}}    # $eq
        ]
    })
    return sem, hyb


def scenario_4_chroma(n=5):
    """
    시나리오 4: 1990년 이전 준공 아파트, 5억 이상 (재건축 기대)
    연산자: $and( $lte, $eq, $gte )
    """
    q = "재건축 투자 가치 높은 오래된 아파트"
    col = get_collection()
    sem = query_chroma(col, q, n_results=n)
    hyb = query_chroma(col, q, n_results=n, where={
        "$and": [
            {"year_built":    {"$lte": 1990}},       # $lte
            {"property_type": {"$eq": "아파트"}},     # $eq
            {"price_eok":     {"$gte": 5.0}}         # $gte
        ]
    })
    return sem, hyb


def scenario_5_chroma(n=5):
    """
    시나리오 5: 단독주택, 주차 가능, 방 3개 이상
    연산자: $and( $eq, $eq, $gte )
    """
    q = "자연 속 조용한 단독주택 마당 텃밭"
    col = get_collection()
    sem = query_chroma(col, q, n_results=n)
    hyb = query_chroma(col, q, n_results=n, where={
        "$and": [
            {"property_type": {"$eq": "단독주택"}},  # $eq
            {"has_parking":   {"$eq": "True"}},      # $eq
            {"rooms":         {"$gte": 3}}           # $gte
        ]
    })
    return sem, hyb


# ══════════════════════════════════════════════════════════════════════════
# Pinecone 하이브리드 검색 시나리오 (3개)
# ══════════════════════════════════════════════════════════════════════════

def scenario_1_pinecone(n=5):
    """
    Pinecone 시나리오 1: 강남구 아파트 15억 이상
    연산자: $and( $eq, $eq, $gte )
    """
    q     = "강남 고급 아파트 역세권 직주근접"
    index = get_pinecone_index()
    sem   = query_pinecone(index, q, n_results=n)
    hyb   = query_pinecone(index, q, n_results=n, filter_dict={
        "$and": [
            {"property_type": {"$eq": "아파트"}},
            {"district":      {"$eq": "강남구"}},
            {"price_eok":     {"$gte": 15.0}}
        ]
    })
    return sem, hyb


def scenario_2_pinecone(n=5):
    """
    Pinecone 시나리오 2: 15층 이상, 20억 이하, 상가 제외
    연산자: $and( $gte, $lte, $ne )
    """
    q     = "한강 조망 고층 아파트 강변 산책"
    index = get_pinecone_index()
    sem   = query_pinecone(index, q, n_results=n)
    hyb   = query_pinecone(index, q, n_results=n, filter_dict={
        "$and": [
            {"floor":         {"$gte": 15}},
            {"price_eok":     {"$lte": 20.0}},
            {"property_type": {"$ne": "상가"}}
        ]
    })
    return sem, hyb


def scenario_3_pinecone(n=5):
    """
    Pinecone 시나리오 3: 5억 이하 OR 오피스텔
    연산자: $or( $lte, $eq )
    """
    q     = "1인 가구 소형 저렴한 오피스텔 역세권"
    index = get_pinecone_index()
    sem   = query_pinecone(index, q, n_results=n)
    hyb   = query_pinecone(index, q, n_results=n, filter_dict={
        "$or": [
            {"price_eok":     {"$lte": 5.0}},
            {"property_type": {"$eq": "오피스텔"}}
        ]
    })
    return sem, hyb


# ══════════════════════════════════════════════════════════════════════════
# 결과 출력 헬퍼
# ══════════════════════════════════════════════════════════════════════════

def print_chroma_results(results: dict, label: str, top_n: int = 5):
    """ChromaDB 검색 결과를 보기 좋게 출력."""
    print(f"\n  ┌{'─'*58}┐")
    print(f"  │ {label:<56} │")
    print(f"  └{'─'*58}┘")

    ids       = results["ids"][0][:top_n]
    metas     = results["metadatas"][0][:top_n]
    distances = results["distances"][0][:top_n]
    docs      = results["documents"][0][:top_n]

    if not ids:
        print("  (결과 없음 — 필터 조건을 완화해보세요)")
        return

    for rank, (rid, meta, dist, doc) in enumerate(zip(ids, metas, distances, docs), 1):
        sim = max(0.0, (1 - dist) * 100)
        print(f"\n  [{rank}] {meta.get('title', rid)}")
        print(f"       유사도: {sim:.1f}%  |  거리: {dist:.4f}")
        print(f"       유형: {meta.get('property_type')} | 구: {meta.get('district')} | "
              f"가격: {meta.get('price_eok')}억 | 면적: {meta.get('area_m2')}㎡ | "
              f"층: {meta.get('floor')}층 | 준공: {meta.get('year_built')}")
        print(f"       {doc[:80]}...")


def print_pinecone_results(results, label: str):
    """Pinecone 검색 결과를 보기 좋게 출력."""
    print(f"\n  ┌{'─'*58}┐")
    print(f"  │ {label:<56} │")
    print(f"  └{'─'*58}┘")

    matches = results.get("matches", [])
    if not matches:
        print("  (결과 없음 — 필터 조건을 완화해보세요)")
        return

    for rank, match in enumerate(matches, 1):
        meta  = match.get("metadata", {})
        score = match.get("score", 0)
        print(f"\n  [{rank}] {meta.get('title', match['id'])}")
        print(f"       유사도 점수: {score:.4f}")
        print(f"       유형: {meta.get('property_type')} | 구: {meta.get('district')} | "
              f"가격: {meta.get('price_eok')}억 | 면적: {meta.get('area_m2')}㎡ | "
              f"층: {meta.get('floor')}층 | 준공: {meta.get('year_built')}")
        desc = meta.get("description", "")
        print(f"       {desc[:80]}...")
