"""
ui/app.py
────────────────────────────────────────────────────────────────────────────
부동산 매물 추천 시스템 - Streamlit UI (확장 기능)

실행:
    streamlit run ui/app.py
────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from db.chroma_client import get_collection
from db.crud import query_chroma

st.set_page_config(page_title="🏠 부동산 매물 추천", layout="wide")

st.title("🏠 부동산 매물 추천 시스템")
st.caption("벡터 DB(ChromaDB) 기반 시맨틱 + 메타데이터 하이브리드 검색")

# ── 사이드바 필터 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 검색 조건")

    query_text = st.text_input(
        "자연어 검색",
        value="역세권 남향 아파트 주차 가능",
        help="원하는 매물을 자연어로 설명하세요"
    )

    st.subheader("메타데이터 필터")

    property_types = st.multiselect(
        "매물 유형",
        ["아파트", "오피스텔", "빌라", "단독주택", "상가"],
        default=[]
    )

    districts = st.multiselect(
        "구(지역)",
        ["강남구", "서초구", "마포구", "용산구", "송파구",
         "성동구", "영등포구", "은평구", "노원구", "강서구",
         "관악구", "동작구", "중구", "종로구", "광진구",
         "중랑구", "강동구", "강북구", "도봉구", "성북구", "서대문구"],
        default=[]
    )

    price_min, price_max = st.slider(
        "가격 범위 (억 원)",
        min_value=1.0, max_value=70.0,
        value=(1.0, 70.0), step=0.5
    )

    rooms_min = st.selectbox("최소 방 수", [0, 1, 2, 3, 4, 5], index=0)

    floor_min = st.number_input("최소 층수", min_value=0, max_value=30, value=0)

    has_parking = st.radio("주차", ["상관없음", "주차 가능만"], index=0)

    n_results = st.slider("검색 결과 수", 1, 20, 5)

    search_btn = st.button("🔍 검색하기", use_container_width=True)

# ── 검색 실행 ─────────────────────────────────────────────────────────────
if search_btn or query_text:
    where_conditions = []

    if property_types:
        if len(property_types) == 1:
            where_conditions.append({"property_type": {"$eq": property_types[0]}})
        else:
            where_conditions.append({"$or": [{"property_type": {"$eq": pt}} for pt in property_types]})

    if districts:
        if len(districts) == 1:
            where_conditions.append({"district": {"$eq": districts[0]}})
        else:
            where_conditions.append({"$or": [{"district": {"$eq": d}} for d in districts]})

    if price_min > 1.0:
        where_conditions.append({"price_eok": {"$gte": price_min}})
    if price_max < 70.0:
        where_conditions.append({"price_eok": {"$lte": price_max}})

    if rooms_min > 0:
        where_conditions.append({"rooms": {"$gte": rooms_min}})

    if floor_min > 0:
        where_conditions.append({"floor": {"$gte": floor_min}})

    if has_parking == "주차 가능만":
        where_conditions.append({"has_parking": {"$eq": "True"}})

    where = None
    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}

    with st.spinner("매물 검색 중..."):
        try:
            collection = get_collection()
            results    = query_chroma(collection, query_text,
                                      n_results=n_results, where=where)

            ids       = results["ids"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]
            docs      = results["documents"][0]

            if not ids:
                st.warning("조건에 맞는 매물이 없습니다. 필터를 완화해 보세요.")
            else:
                st.success(f"**{len(ids)}개** 매물을 찾았습니다.")
                if where:
                    st.caption(f"적용된 필터: {where}")

                for rank, (rid, meta, dist, doc) in enumerate(
                        zip(ids, metas, distances, docs), 1):
                    sim_pct = max(0, (1 - dist) * 100)
                    with st.expander(
                        f"#{rank} {meta.get('title', rid)}  |  "
                        f"유사도 {sim_pct:.1f}%  |  "
                        f"{meta.get('price_eok')}억",
                        expanded=(rank <= 3)
                    ):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("유형",  meta.get("property_type", "-"))
                        col1.metric("구",    meta.get("district", "-"))
                        col2.metric("가격",  f"{meta.get('price_eok')}억")
                        col2.metric("면적",  f"{meta.get('area_m2')}㎡")
                        col3.metric("층수",  f"{meta.get('floor')}층")
                        col3.metric("방수",  f"{meta.get('rooms')}개")

                        parking = "✅ 가능" if meta.get("has_parking") == "True" else "❌ 불가"
                        st.write(f"**주차:** {parking} | **준공:** {meta.get('year_built')}년")
                        st.write(f"**설명:** {doc}")
                        st.progress(sim_pct / 100,
                                    text=f"시맨틱 유사도: {sim_pct:.1f}%")

        except Exception as e:
            st.error(f"검색 오류: {e}")
            st.info("먼저 `python init_db.py` 를 실행해 DB를 초기화하세요.")
