"""
db/pinecone_client.py
────────────────────────────────────────────────────────────────────────────
Pinecone 서버리스 인덱스 클라이언트

핵심 설계 결정:
  - 서버리스 인덱스: AWS us-east-1, cosine 유사도
  - dimension=1536: text-embedding-3-small 고정 차원을 명시 (요건 충족)
  - 외부 임베딩 주입 방식: 벡터를 직접 생성해서 Pinecone에 전달
    (ChromaDB의 "텍스트를 주면 벡터로 만들어주는" 방식과 대비됨)
────────────────────────────────────────────────────────────────────────────
"""

import os
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME      = "real-estate"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536          # text-embedding-3-small 고정 차원 (명시 필수)
BATCH_SIZE      = 50


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_pinecone_index():
    """
    Pinecone 인덱스를 반환. 없으면 서버리스 인덱스를 생성.
    인덱스 생성 시 dimension을 정확히 명시 (과제 요건).
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"[Pinecone] 인덱스 '{INDEX_NAME}' 생성 중...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,       # text-embedding-3-small 차원 명시
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # 인덱스가 Ready 상태가 될 때까지 대기
        while True:
            info = pc.describe_index(INDEX_NAME)
            if info.status.get("ready", False):
                break
            print("[Pinecone] 인덱스 준비 중... 5초 대기")
            time.sleep(5)
        print(f"[Pinecone] 인덱스 '{INDEX_NAME}' 생성 완료.")

    return pc.Index(INDEX_NAME)


def embed_texts(texts: list[str], openai_client: OpenAI = None) -> list[list[float]]:
    """
    텍스트 리스트를 OpenAI API로 임베딩하여 벡터 리스트를 반환.
    Pinecone은 벡터를 직접 받으므로, 여기서 임베딩을 생성해 주입함
    (ChromaDB의 자동 임베딩 방식과 다른 "외부 주입" 방식).
    """
    if openai_client is None:
        openai_client = get_openai_client()

    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]
