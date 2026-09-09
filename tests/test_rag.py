from pathlib import Path

from gaode.rag.retriever import (
    _merge_documents,
    _parse_metadata,
    _search_local_documents,
    documents_to_sources,
)
from langchain_core.documents import Document


def test_markdown_metadata_is_structured() -> None:
    raw = """# 故宫博物院
- 地址：北京市东城区景山前街4号
- 评分：4.8
- 类型：历史文化
- 门票：60元
- 开放时间：08:30-17:00
"""
    metadata = _parse_metadata(Path("故宫博物院.md"), raw, "attraction")
    assert metadata["name"] == "故宫博物院"
    assert metadata["city"] == "北京"
    assert metadata["rating"] == 4.8
    assert metadata["ticket_price"] == "60元"


def test_city_detection_is_not_limited_to_seed_cities() -> None:
    raw = """# 鼓浪屿
- 城市：厦门市
- 地址：福建省厦门市思明区
- 评分：4.7
- 类型：历史文化
"""
    metadata = _parse_metadata(Path("鼓浪屿.md"), raw, "attraction")
    assert metadata["city"] == "厦门"


def test_vector_result_keeps_priority_and_sources_are_deduplicated() -> None:
    vector = Document(
        page_content="故宫资料",
        metadata={
            "source": "resource/knowledge/attractions/故宫博物院.md",
            "name": "故宫博物院",
            "city": "北京",
            "type": "attraction",
            "score": 0.91,
        },
    )
    duplicate = Document(page_content="重复", metadata={"source": vector.metadata["source"]})
    merged = _merge_documents([vector], [duplicate], 5)
    sources = documents_to_sources(merged)
    assert len(merged) == 1
    assert len(sources) == 1
    assert sources[0].title == "故宫博物院"
    assert sources[0].score == 0.91


def test_retrieval_never_backfills_from_another_city() -> None:
    documents = _search_local_documents("热门景点", "attraction", city="成都", k=4)
    assert documents
    assert {document.metadata["city"] for document in documents} == {"成都"}


def test_same_knowledge_item_is_deduplicated_across_source_path_styles() -> None:
    vector = Document(
        page_content="向量结果",
        metadata={"source": "resource/x.md", "name": "西湖", "city": "杭州", "type": "attraction"},
    )
    local = Document(
        page_content="本地结果",
        metadata={
            "source": "C:/repo/resource/x.md",
            "name": "西湖",
            "city": "杭州",
            "kind": "attraction",
        },
    )
    assert len(_merge_documents([vector], [local], 4)) == 1
