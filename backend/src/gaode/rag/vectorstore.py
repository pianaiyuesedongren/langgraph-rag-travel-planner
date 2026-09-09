from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from gaode.config import Settings, get_settings


class VectorStoreManager:
    """Milvus Lite向量存储管理器 (WSL2/Linux环境)"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        # Bump the collection when the stored schema changes. This avoids mixing
        # legacy rows that do not contain the structured travel metadata.
        self._collection_name = "travel_knowledge_v3"
        self._client = None

    def _get_client(self):
        """获取Milvus客户端"""
        if self._client is not None:
            return self._client

        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "Missing pymilvus. Install: pip install pymilvus[milvus_lite]"
            ) from exc

        uri = str(Path(__file__).resolve().parents[4] / "backend" / "milvus" / "travel.db")
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        self._client = MilvusClient(uri=uri)
        if self._client.has_collection(self._collection_name):
            self._client.load_collection(self._collection_name)
        return self._client

    def _ensure_collection(self, dimension: int = 1024) -> None:
        """确保集合存在"""
        client = self._get_client()

        if client.has_collection(self._collection_name):
            client.load_collection(self._collection_name)
            return

        client.create_collection(
            collection_name=self._collection_name,
            dimension=dimension,
            primary_field_name="id",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
            enable_dynamic_field=True,
        )
        client.load_collection(self._collection_name)

    def _row_count(self) -> int:
        client = self._get_client()
        if not client.has_collection(self._collection_name):
            return 0
        try:
            stats = client.get_collection_stats(self._collection_name)
            return int(stats.get("row_count", 0) or 0)
        except Exception:
            return 0

    def add_documents(
        self,
        documents: list[Document],
        embeddings: Embeddings,
    ) -> int:
        """添加文档到向量存储"""
        if not documents:
            return 0

        client = self._get_client()
        valid_docs = [
            doc
            for doc in documents
            if doc.page_content is not None and str(doc.page_content).strip()
        ]
        texts = [str(doc.page_content).strip() for doc in valid_docs]

        if not texts:
            return 0

        vectors = embeddings.embed_documents(texts)

        if vectors:
            self._ensure_collection(dimension=len(vectors[0]))

        data = []
        if len(vectors) != len(valid_docs):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(valid_docs)} documents"
            )

        for i, (doc, vector) in enumerate(zip(valid_docs, vectors, strict=True)):
            source = str(doc.metadata.get("source", ""))
            source_path = Path(source)
            project_root = Path(__file__).resolve().parents[4]
            try:
                source = source_path.resolve().relative_to(project_root).as_posix()
            except (OSError, ValueError):
                source = source_path.as_posix()
            identity = "|".join(
                [
                    source,
                    str(doc.metadata.get("chunk_index", i)),
                    str(doc.page_content),
                ]
            )
            # Keep ingestion idempotent: importing the same directory again
            # updates the same Milvus rows instead of creating duplicates.
            stable_id = int.from_bytes(
                hashlib.sha1(identity.encode("utf-8")).digest()[:8],
                byteorder="big",
                signed=False,
            ) & ((1 << 63) - 1)
            data.append(
                {
                    "id": stable_id,
                    "vector": vector,
                    "content": str(doc.page_content),
                    "source": source,
                    "type": doc.metadata.get("type", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "name": doc.metadata.get("name", ""),
                    "city": doc.metadata.get("city", ""),
                    "address": doc.metadata.get("address", ""),
                    "rating": float(doc.metadata.get("rating", 0.0) or 0.0),
                    "category": doc.metadata.get("category", ""),
                    "summary": doc.metadata.get("summary", ""),
                    "source_url": doc.metadata.get("source_url", ""),
                    "verified_at": doc.metadata.get("verified_at", ""),
                    "avg_price": float(doc.metadata.get("avg_price", 0.0) or 0.0),
                    "tags": list(doc.metadata.get("tags", []) or []),
                    "business_hours": doc.metadata.get("business_hours", ""),
                    "ticket_price": doc.metadata.get("ticket_price", ""),
                    "elderly_friendly": bool(doc.metadata.get("elderly_friendly", False)),
                    "child_friendly": bool(doc.metadata.get("child_friendly", False)),
                }
            )

        client.upsert(collection_name=self._collection_name, data=data)
        return len(data)

    def ensure_seeded(self, documents: list[Document], embeddings: Embeddings) -> int:
        """Create and seed the collection when it is empty."""
        if not documents:
            return 0
        if self._row_count() > 0:
            return 0
        return self.add_documents(documents, embeddings)

    def search(
        self,
        query: str,
        embeddings: Embeddings,
        k: int = 5,
        filter_type: str | None = None,
    ) -> list[Document]:
        """搜索相似文档"""
        client = self._get_client()

        if not client.has_collection(self._collection_name):
            return []

        query_vector = embeddings.embed_query(query)

        search_params = {"metric_type": "COSINE", "params": {}}
        filter_expr = f'type == "{filter_type}"' if filter_type else None

        results = client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            limit=k,
            output_fields=[
                "content",
                "source",
                "type",
                "chunk_index",
                "name",
                "city",
                "address",
                "rating",
                "category",
                "summary",
                "source_url",
                "verified_at",
                "avg_price",
                "tags",
                "business_hours",
                "ticket_price",
                "elderly_friendly",
                "child_friendly",
            ],
            search_params=search_params,
            filter=filter_expr,
        )

        documents = []
        for item in results[0]:
            entity = item.get("entity", {})
            documents.append(
                Document(
                    page_content=entity.get("content", ""),
                    metadata={
                        "source": entity.get("source", ""),
                        "type": entity.get("type", ""),
                        "chunk_index": entity.get("chunk_index", 0),
                        "score": item.get("distance", 0),
                        "name": entity.get("name", ""),
                        "city": entity.get("city", ""),
                        "address": entity.get("address", ""),
                        "rating": entity.get("rating", 0.0),
                        "category": entity.get("category", ""),
                        "summary": entity.get("summary", ""),
                        "source_url": entity.get("source_url", ""),
                        "verified_at": entity.get("verified_at", ""),
                        "avg_price": entity.get("avg_price", 0.0),
                        "tags": entity.get("tags", []),
                        "business_hours": entity.get("business_hours", ""),
                        "ticket_price": entity.get("ticket_price", ""),
                        "elderly_friendly": entity.get("elderly_friendly", False),
                        "child_friendly": entity.get("child_friendly", False),
                    },
                )
            )

        return documents

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计"""
        return {"total_documents": self._row_count()}


@lru_cache(maxsize=1)
def get_vectorstore_manager(settings: Settings | None = None) -> VectorStoreManager:
    """Reuse one Milvus Lite client per API process."""
    return VectorStoreManager(settings or get_settings())
