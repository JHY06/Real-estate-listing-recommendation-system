"""
benchmark/compare.py
────────────────────────────────────────────────────────────────────────────
파일 기반(JSON + numpy 코사인 유사도) vs 벡터 DB(ChromaDB) 성능/기능 비교

[비교 차원 1] 검색 속도 (정량)
  - 데이터 규모: 100 / 150 / 200 / 250개
  - 각 규모에서 동일 쿼리 5회 반복 → 평균 시간(ms) 측정
  - 결과 그래프 저장: ./benchmark/speed_comparison.png

[비교 차원 2] 기능 비교 (정성)
  - 메타데이터 필터링 방식
  - 영속성(재시작 후 데이터 유지) 처리
  - 코드 복잡도 / 확장성
────────────────────────────────────────────────────────────────────────────
"""

import time
import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = "./benchmark/embeddings_cache.pkl"


# ══════════════════════════════════════════════════════════════════════════
# 파일 기반 시스템
# ══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def build_file_store(properties: list[dict], openai_client: OpenAI) -> dict:
    """
    파일 기반 임베딩 스토어 구축.
    캐시 존재 시 재사용 (불필요한 API 비용 방지).
    """
    if os.path.exists(CACHE_PATH):
        print("[파일 기반] 캐시에서 임베딩 로드...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("[파일 기반] OpenAI로 임베딩 생성 중...")
    store      = {}
    texts      = [p["title"] + ". " + p["description"] for p in properties]
    ids        = [p["id"] for p in properties]
    batch_size = 50
    embeddings = []

    for i in range(0, len(texts), batch_size):
        resp = openai_client.embeddings.create(
            input=texts[i:i+batch_size], model="text-embedding-3-small"
        )
        embeddings.extend([item.embedding for item in resp.data])

    for pid, emb, prop in zip(ids, embeddings, properties):
        store[pid] = {"embedding": np.array(emb), "metadata": prop}

    os.makedirs("./benchmark", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(store, f)
    print(f"[파일 기반] {len(store)}개 임베딩 생성 및 캐시 저장 완료.")
    return store


def file_based_search(store: dict, query_embedding: np.ndarray, n: int = 5):
    """파일 기반 브루트포스 O(n) 코사인 유사도 검색."""
    scores = [
        (pid, cosine_similarity(query_embedding, item["embedding"]))
        for pid, item in store.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# ══════════════════════════════════════════════════════════════════════════
# 벤치마크 실행
# ══════════════════════════════════════════════════════════════════════════

def run_benchmark():
    """
    [비교 차원 1] 검색 속도 정량 비교
    - 데이터 규모: 100 / 150 / 200 / 250개
    - 각 규모에서 5회 반복 측정 후 평균(ms)
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.properties import PROPERTIES
    from db.chroma_client import get_collection

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    collection    = get_collection()

    QUERY = "한강 조망 역세권 아파트 주차 가능"
    SIZES = [100, 150, 200, 250]
    REPS  = 5

    # 쿼리 임베딩 (한 번만 생성)
    q_resp      = openai_client.embeddings.create(
        input=[QUERY], model="text-embedding-3-small"
    )
    q_embedding = np.array(q_resp.data[0].embedding)

    # 전체 파일 기반 스토어 구축 (캐시 사용)
    full_store = build_file_store(PROPERTIES, openai_client)
    all_ids    = [p["id"] for p in PROPERTIES]

    file_times   = []
    chroma_times = []

    print("\n[비교 차원 1] 검색 속도 측정 시작...")
    for size in SIZES:
        subset_ids   = all_ids[:size]
        subset_store = {k: full_store[k] for k in subset_ids if k in full_store}

        # 파일 기반 검색 속도
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            file_based_search(subset_store, q_embedding, n=5)
            times.append(time.perf_counter() - t0)
        avg_file = sum(times) / REPS * 1000
        file_times.append(avg_file)

        # ChromaDB 검색 속도
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            collection.query(
                query_texts=[QUERY],
                n_results=5,
                include=["metadatas", "distances"]
            )
            times.append(time.perf_counter() - t0)
        avg_chroma = sum(times) / REPS * 1000
        chroma_times.append(avg_chroma)

        print(f"  크기 {size:>3}개 | 파일 기반: {avg_file:7.2f}ms | "
              f"ChromaDB: {avg_chroma:7.2f}ms | "
              f"비율: {avg_file/avg_chroma:.1f}x 빠름")

    _plot_benchmark(SIZES, file_times, chroma_times)
    return SIZES, file_times, chroma_times


def _plot_benchmark(sizes, file_times, chroma_times):
    """검색 속도 비교 그래프 저장."""
    # 한글 폰트 설정
    for fname in ["NanumGothic", "AppleGothic", "Malgun Gothic", "DejaVu Sans"]:
        try:
            fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
            plt.rcParams["font.family"] = fname
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 그래프 1: 속도 비교 꺾은선 ────────────────────────────────────────
    ax = axes[0]
    ax.plot(sizes, file_times,   "o-", color="#E74C3C", linewidth=2,
            markersize=8, label="File-based (numpy O(n))")
    ax.plot(sizes, chroma_times, "s-", color="#2ECC71", linewidth=2,
            markersize=8, label="ChromaDB (HNSW ANN)")
    ax.set_xlabel("Data Size", fontsize=12)
    ax.set_ylabel("Avg Query Time (ms)", fontsize=12)
    ax.set_title("Search Speed: File-based vs ChromaDB", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    for x, y in zip(sizes, file_times):
        ax.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#E74C3C")
    for x, y in zip(sizes, chroma_times):
        ax.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9, color="#2ECC71")

    # ── 그래프 2: 속도 비율 막대 ───────────────────────────────────────────
    ax2   = axes[1]
    ratios = [f / c for f, c in zip(file_times, chroma_times)]
    bars   = ax2.bar([str(s) for s in sizes], ratios,
                     color=["#3498DB", "#9B59B6", "#E67E22", "#E74C3C"],
                     edgecolor="white", linewidth=1.2)
    ax2.set_xlabel("Data Size", fontsize=12)
    ax2.set_ylabel("Speed Ratio (File / ChromaDB)", fontsize=12)
    ax2.set_title("How Many Times Faster is ChromaDB?", fontsize=13, fontweight="bold")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Equal speed")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, ratio in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{ratio:.1f}x", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    os.makedirs("./benchmark", exist_ok=True)
    path = "./benchmark/speed_comparison.png"
    plt.suptitle("File-based vs ChromaDB Vector DB — Performance Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[벤치마크] 그래프 저장 완료: {path}")


if __name__ == "__main__":
    run_benchmark()
