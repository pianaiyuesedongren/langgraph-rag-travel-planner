from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """文本分块器 - 将长文档分割成适合向量化的片段"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """分割文档列表"""
        if not documents:
            return []

        split_docs = self._splitter.split_documents(documents)

        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_index"] = i

        return split_docs

    def split_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """分割单个文本"""
        chunks = self._splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            doc_metadata = (metadata or {}).copy()
            doc_metadata["chunk_index"] = i
            documents.append(
                Document(
                    page_content=chunk,
                    metadata=doc_metadata,
                )
            )

        return documents
