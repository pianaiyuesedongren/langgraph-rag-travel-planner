from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from gaode.config import Settings, get_settings
from gaode.llm.client import LocalHashEmbeddings, get_embeddings
from gaode.rag.splitter import TextSplitter
from gaode.rag.vectorstore import get_vectorstore_manager
from gaode.schemas.travel import EvidenceSource

KNOWN_CITIES = ("北京", "上海", "天津", "重庆", "杭州", "成都", "西安")


def _knowledge_root() -> Path:
    return Path(__file__).resolve().parents[4] / "resource" / "knowledge"


@lru_cache(maxsize=1)
def _bootstrap_vectorstore() -> bool:
    settings = get_settings()
    if not settings.vector_store_enabled:
        return False

    root = _knowledge_root()
    if not root.exists():
        return False

    documents = list(_local_documents("attraction")) + list(_local_documents("restaurant"))
    if not documents:
        return False

    split_docs = TextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    split_docs = [doc for doc in split_docs if str(doc.page_content).strip()]
    if not split_docs:
        return False

    vectorstore = get_vectorstore_manager(settings)
    if vectorstore.get_stats().get("total_documents", 0) > 0:
        return True

    try:
        embeddings = get_embeddings(settings)
    except Exception:
        embeddings = LocalHashEmbeddings()

    try:
        return vectorstore.ensure_seeded(split_docs, embeddings) > 0
    except Exception:
        return False


def _detect_city(text: str) -> str:
    for city in KNOWN_CITIES:
        if city in text:
            return city
    match = re.search(r"([\u4e00-\u9fff]{2,8}?)(?:市|自治州|地区)", text)
    if match:
        return match.group(1)
    return ""


def _first_match(patterns: tuple[str, ...], text: str, default: str = "") -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return default


def _parse_metadata(path: Path, raw: str, kind: str) -> dict[str, object]:
    title = _first_match((r"^#\s*(.+)$",), raw, default=path.stem)
    address = _first_match((r"^\s*[-*]?\s*(?:地址|地点|位置)[:：]\s*(.+)$",), raw)
    city = _first_match((r"^\s*[-*]?\s*(?:城市|所在城市)[:：]\s*(.+)$",), raw)
    rating = _first_match((r"^\s*[-*]?\s*(?:评分|評分)[:：]\s*([0-9.]+)",), raw)
    category = _first_match((r"^\s*[-*]?\s*(?:类型|類型)[:：]\s*(.+)$",), raw)
    avg_price = _first_match((r"^\s*[-*]?\s*(?:人均|均价)[:：]\s*([0-9.]+)",), raw)
    source_url = _first_match((r"^\s*[-*]?\s*(?:来源|来源网址)[:：]\s*(.+)$",), raw)
    verified_at = _first_match((r"^\s*[-*]?\s*(?:核验日期|更新日期)[:：]\s*(.+)$",), raw)

    metadata: dict[str, object] = {
        "name": title,
        "source": str(path),
        "city": city.removesuffix("市") or _detect_city(address or raw or title),
        "address": address,
        "rating": float(rating or 0.0),
        "category": category,
        "kind": kind,
        "summary": raw.strip().splitlines()[0][:400] if raw.strip() else "",
        "source_url": source_url,
        "verified_at": verified_at,
    }

    if kind == "restaurant":
        metadata["avg_price"] = float(avg_price or 0.0)
        metadata["tags"] = [
            line.strip("- ").strip() for line in raw.splitlines() if "：" in line or ":" in line
        ]
    else:
        metadata["business_hours"] = _first_match(
            (r"^\s*[-*]?\s*(?:营业时间|營業時間|开放时间|開放時間)[:：]\s*(.+)$",), raw
        )
        metadata["ticket_price"] = _first_match((r"^\s*[-*]?\s*(?:门票|門票)[:：]\s*(.+)$",), raw)
        metadata["elderly_friendly"] = "适老" in raw or "老人" in raw
        metadata["child_friendly"] = "亲子" in raw or "儿童" in raw or "小孩" in raw
    return metadata


@lru_cache(maxsize=2)
def _local_documents(kind: str) -> tuple[Document, ...]:
    folder = _knowledge_root() / ("restaurants" if kind == "restaurant" else "attractions")
    if not folder.exists():
        return ()

    docs: list[Document] = []
    for path in sorted(folder.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        docs.append(Document(page_content=raw, metadata=_parse_metadata(path, raw, kind)))
    return tuple(docs)


def _score_local_document(query: str, city: str, doc: Document, kind: str) -> float:
    text = f"{doc.metadata.get('name', '')}\n{doc.page_content}".lower()
    query_lower = query.lower()
    score = float(doc.metadata.get("rating", 0.0)) * 10.0

    doc_city = str(doc.metadata.get("city", ""))
    if city and doc_city == city:
        score += 80.0
    elif city and city in text:
        score += 40.0

    name = str(doc.metadata.get("name", "")).lower()
    if name and name in query_lower:
        score += 20.0
    if any(token in query_lower for token in ("经典", "热门", "推荐", "景点", "餐厅", "美食")):
        score += 10.0
    if kind == "attraction" and "特色景点" in doc.page_content:
        score += 4.0
    if kind == "restaurant" and "特色菜品" in doc.page_content:
        score += 4.0
    return score


def _search_local_documents(query: str, kind: str, city: str = "", k: int = 5) -> list[Document]:
    docs = list(_local_documents(kind))
    if not docs:
        return []

    ranked = sorted(
        docs,
        key=lambda doc: (
            _score_local_document(query, city, doc, kind),
            float(doc.metadata.get("rating", 0.0)),
            str(doc.metadata.get("name", "")),
        ),
        reverse=True,
    )
    return ranked[: max(1, k)]


def _merge_documents(primary: list[Document], secondary: list[Document], k: int) -> list[Document]:
    merged: list[Document] = []
    seen: set[str] = set()

    for doc in primary + secondary:
        metadata = getattr(doc, "metadata", {}) or {}
        key = str(metadata.get("source") or metadata.get("name") or doc.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= k:
            break

    return merged


def documents_to_sources(documents: list[Document]) -> list[EvidenceSource]:
    sources: list[EvidenceSource] = []
    seen: set[str] = set()
    for document in documents:
        metadata = document.metadata or {}
        source = str(metadata.get("source", ""))
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(
            EvidenceSource(
                source=source,
                url=str(metadata.get("source_url", "")),
                title=str(metadata.get("name", "")),
                kind=str(metadata.get("type") or metadata.get("kind") or "general"),
                city=str(metadata.get("city", "")),
                score=float(metadata.get("score", 0.0) or 0.0),
            )
        )
    return sources


class TravelRetriever:
    """Travel knowledge retriever with Milvus + markdown fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._vectorstore = get_vectorstore_manager(self.settings)
        self._embeddings = None
        _bootstrap_vectorstore()

    def _get_embeddings(self) -> Embeddings:
        if self._embeddings is not None:
            return self._embeddings

        from gaode.llm import get_embeddings

        self._embeddings = get_embeddings(self.settings)
        return self._embeddings

    def _vector_search(
        self,
        query: str,
        kind: str,
        filter_type: str | None,
        city: str,
        k: int,
    ) -> list[Document]:
        embeddings = self._get_embeddings()
        docs = self._vectorstore.search(
            query=query,
            embeddings=embeddings,
            k=k,
            filter_type=filter_type,
        )
        local_docs = _search_local_documents(query, kind, city=city, k=k)
        if docs:
            return _merge_documents(docs, local_docs, k)

        if filter_type is not None:
            docs = self._vectorstore.search(
                query=query,
                embeddings=embeddings,
                k=k,
            )
        if docs:
            return _merge_documents(docs, local_docs, k)
        return local_docs

    def retrieve_attractions(
        self,
        query: str,
        city: str = "",
        k: int = 5,
    ) -> list[Document]:
        enhanced_query = f"{city} 景点 {query}" if city else f"景点 {query}"
        try:
            return self._vector_search(
                query=enhanced_query,
                kind="attraction",
                filter_type="attraction",
                city=city,
                k=k,
            )
        except Exception:
            return _search_local_documents(enhanced_query, "attraction", city=city, k=k)

    def retrieve_restaurants(
        self,
        query: str,
        city: str = "",
        k: int = 5,
    ) -> list[Document]:
        enhanced_query = f"{city} 餐厅 美食 {query}" if city else f"餐厅 美食 {query}"
        try:
            return self._vector_search(
                query=enhanced_query,
                kind="restaurant",
                filter_type="restaurant",
                city=city,
                k=k,
            )
        except Exception:
            return _search_local_documents(enhanced_query, "restaurant", city=city, k=k)

    def retrieve_general(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        try:
            embeddings = self._get_embeddings()
            return self._vectorstore.search(
                query=query,
                embeddings=embeddings,
                k=k,
            )
        except Exception:
            return _search_local_documents(query, "attraction", k=k)

    def get_knowledge_stats(self) -> dict[str, Any]:
        try:
            return self._vectorstore.get_stats()
        except Exception:
            return {
                "total_documents": len(_local_documents("attraction"))
                + len(_local_documents("restaurant"))
            }
