#!/usr/bin/env python3
"""文档导入脚本 - 将文档导入到Milvus向量数据库"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend" / "src"))

from gaode.config import get_settings
from gaode.llm.client import get_embeddings
from gaode.rag.loader import DocumentLoader
from gaode.rag.retriever import _parse_metadata
from gaode.rag.splitter import TextSplitter
from gaode.rag.vectorstore import VectorStoreManager


def ingest_directory(directory: str, doc_type: str = "general") -> int:
    """导入指定目录的文档"""
    settings = get_settings()
    embeddings = get_embeddings(settings)
    print(f"Using embedding provider: {type(embeddings).__name__}")

    loader = DocumentLoader(directory)
    documents = loader.load()

    if not documents:
        print(f"No documents found in {directory}")
        return 0

    for doc in documents:
        source = Path(str(doc.metadata.get("source", "")))
        doc.metadata.update(_parse_metadata(source, str(doc.page_content), doc_type))
        doc.metadata["type"] = doc_type
        doc.page_content = str(doc.page_content) if doc.page_content else ""

    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    split_docs = [doc for doc in split_docs if doc.page_content and doc.page_content.strip()]

    print(f"Loaded {len(documents)} documents, split into {len(split_docs)} chunks")

    if not split_docs:
        return 0

    vectorstore = VectorStoreManager(settings)
    added = vectorstore.add_documents(split_docs, embeddings)

    print(f"Added {added} chunks to vector store")
    return added


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("--dir", required=True, help="Directory containing documents")
    parser.add_argument("--type", default="general", help="Document type")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    count = ingest_directory(str(target_dir), args.type)
    print(f"\nDone! Added {count} chunks")


if __name__ == "__main__":
    main()
