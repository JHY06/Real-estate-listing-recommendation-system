"""
db/chroma_client.py
────────────────────────────────────────────────────────────────────────────
ChromaDB PersistentClient + OpenAI Custom Embedding Function

핵심 설계 결정:
  - PersistentClient: 디스크에 영구 저장 → 프로그램 재실행 시 재임베딩 불필요
  - CustomOpenAIEmbeddingFunction: 기본 SentenceTransformer 대신
    OpenAI text-embedding-3-small 사용 (확장 기능 요건)
  - ID 기반 중복 체크: 이미 존재하는 항목은 임베딩 API 호출 생략
────────────────────────────────────────────────────────────────────────────
"""

import os
import chromadb
from chromadb import EmbeddingFunction, Embeddings
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI 임베딩 모델 설정 ────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536        # text-embedding-3-small 고정 차원
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "real_estate"
BATCH_SIZE      = 50          # OpenAI API 배치 요청 단위


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB용 커스텀 임베딩 함수 (확장 기능 A)
    ChromaDB의 기본 SentenceTransformer 대신 OpenAI API 사용.
    EmbeddingFunction 인터페이스를 구현하여 컬렉션에 직접 등록.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = model

    def __call__(self, input: List[str]) -> Embeddings:
        """ChromaDB가 내부적으로 호출하는 메서드."""
        response = self.client.embeddings.create(
            input=input,
            model=self.model
        )
        return [item.embedding for item in response.data]


def get_chroma_client() -> chromadb.PersistentClient:
    """
    PersistentClient 반환.
    CHROMA_DIR 경로에 데이터가 영구 저장됨 → 재실행 시 데이터 유지.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client: chromadb.PersistentClient = None,
                   name: str = COLLECTION_NAME):
    """
    컬렉션을 가져오거나 없으면 생성.
    get_or_create_collection: 동일 이름 컬렉션이 이미 존재해도 오류 없음(멱등).
    """
    if client is None:
        client = get_chroma_client()
    embedding_fn = OpenAIEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}   # 코사인 유사도 사용
    )
    return collection
