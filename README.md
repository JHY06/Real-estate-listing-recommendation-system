# 🏠 부동산 매물 추천 시스템
> 벡터 데이터베이스(ChromaDB + Pinecone) 기반 지능형 매물 추천 시스템

---

## 📁 프로젝트 구조

```
real_estate_vector_db/
├── .env                    ← API 키 (gitignore 등록)
├── .env.example
├── requirements.txt
├── init_db.py              ← DB 초기화 (ChromaDB / Pinecone)
├── demo.py                 ← 전체 기능 데모 (CRUD + 검색 + 벤치마크)
├── data/
│   └── properties.py       ← 250개 서울 부동산 매물 데이터셋
├── db/
│   ├── chroma_client.py    ← ChromaDB PersistentClient + 커스텀 임베딩 함수
│   ├── pinecone_client.py  ← Pinecone 서버리스 인덱스
│   └── crud.py             ← ChromaDB/Pinecone CRUD 전체
├── search/
│   └── hybrid_search.py    ← 하이브리드 검색 5개 시나리오
├── benchmark/
│   └── compare.py          ← 파일 기반 vs ChromaDB 속도 비교
└── ui/
    └── app.py              ← Streamlit 웹 UI (확장 기능)
```

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 입력
```

### 2. DB 초기화 (데이터 적재)
```bash
# ChromaDB만
python init_db.py --db chroma

# Pinecone만
python init_db.py --db pinecone

# 둘 다
python init_db.py --db both

# 다중 컬렉션 (property_type별 분리, 확장 기능)
python init_db.py --db chroma --multi

# 리셋 후 재삽입
python init_db.py --db both --reset
```

### 3. 전체 데모 실행
```bash
# 전체 (CRUD + 검색 + 벤치마크)
python demo.py

# 단계별 실행
python demo.py --step crud
python demo.py --step search
python demo.py --step benchmark
```

### 4. Streamlit UI
```bash
streamlit run ui/app.py
```

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 총 매물 수 | **250개** |
| 커버 지역 | 서울 21개 자치구 |
| 매물 유형 | 아파트 / 오피스텔 / 빌라 / 단독주택 / 상가 |
| 임베딩 모델 | OpenAI `text-embedding-3-small` (dim=1536) |

**메타데이터 필드 (8개)**

| 필드 | 타입 | 설명 |
|------|------|------|
| `property_type` | str | 아파트/오피스텔/빌라/단독주택/상가 |
| `district` | str | 서울 자치구 |
| `price_eok` | float | 매매가 (억 원) |
| `area_m2` | float | 전용면적 (㎡) |
| `rooms` | int | 방 수 |
| `year_built` | int | 준공연도 |
| `floor` | int | 층수 |
| `has_parking` | bool | 주차 가능 여부 |

---

## 🔧 핵심 구현 사항

### A. ChromaDB vs Pinecone 임베딩 방식 비교

| | ChromaDB | Pinecone |
|--|---------|---------|
| 임베딩 방식 | `EmbeddingFunction` 등록 → **자동 벡터화** | 외부에서 벡터 생성 후 **직접 주입** |
| 텍스트 입력 | `collection.add(documents=[...])` | `embed_texts()` → `index.upsert(vectors=[...])` |
| 장점 | 편리성 ↑ | 유연성 ↑ (다양한 임베딩 모델 혼용 가능) |

### B. UPDATE vs UPSERT (ChromaDB)

```python
# UPDATE: 반드시 존재해야 수정 가능 → 없으면 예외 발생
collection.update(ids=["p001"], metadatas=[{...}])

# UPSERT: 없으면 생성, 있으면 덮어씀 → 멱등성 보장
collection.upsert(ids=["p001"], documents=["..."], metadatas=[{...}])
```

**멱등성(Idempotency)**: 동일한 upsert를 여러 번 실행해도 결과가 항상 동일.

### C. 하이브리드 검색 연산자

| 시나리오 | 연산자 조합 |
|---------|-----------|
| 강남구 아파트 15억↑ | `$and`: `$eq`, `$eq`, `$gte` |
| 고층 한강뷰 20억↓ | `$and`: `$gte`, `$lte`, `$ne` |
| 소형/저가 매물 | `$or`: `$lte`, `$eq` |
| 재건축 기대 매물 | `$and`: `$lte`, `$eq`, `$gte` |
| 자연 인접 단독주택 | `$and`: `$eq`, `$eq`, `$gte` |

### D. 파일 기반 vs 벡터 DB 비교

| 비교 항목 | 파일 기반 (numpy) | ChromaDB |
|---------|-----------------|---------|
| 검색 복잡도 | O(n) 브루트포스 | O(log n) HNSW |
| 메타데이터 필터 | 수동 Python 코드 | `where={}` 파라미터 |
| 영속성 | pickle 직접 저장 | PersistentClient 자동 |
| 코드 복잡도 | 높음 | 낮음 |

---

## 🌟 확장 기능

1. **커스텀 임베딩 함수** (`db/chroma_client.py`)
   - ChromaDB 기본 SentenceTransformer 대신 OpenAI `text-embedding-3-small` 사용
   - `EmbeddingFunction` 인터페이스 직접 구현

2. **다중 컬렉션** (`init_db.py --multi`)
   - property_type별 별도 컬렉션 분리
   - 멀티테넌시/도메인 분리 아키텍처 시연

3. **Streamlit UI** (`ui/app.py`)
   - 자연어 검색 + 사이드바 메타데이터 필터
   - 시맨틱 유사도 시각화

---

## 📈 기술 스택

- **벡터 DB**: ChromaDB (PersistentClient, HNSW, cosine) + Pinecone (Serverless, AWS us-east-1)
- **임베딩**: OpenAI `text-embedding-3-small` (dim=1536)
- **UI**: Streamlit
- **벤치마크**: numpy, matplotlib
