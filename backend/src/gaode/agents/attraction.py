from __future__ import annotations

import asyncio
import json
import os

from langchain_core.messages import AIMessage, HumanMessage

from gaode.agents.state import AgentState, agent_timeout_seconds
from gaode.llm import get_chat_model
from gaode.rag.retriever import TravelRetriever, documents_to_sources
from gaode.schemas.travel import Attraction, TraceEvent
from gaode.tools.mcp_registry import MCPToolRegistry


def _trace(step: str, message: str, **data) -> TraceEvent:
    return TraceEvent(step=step, message=message, data=data)


def _first_user_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    message = next(
        (item for item in reversed(messages) if isinstance(item, HumanMessage)),
        messages[-1],
    )
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content or "")


def _react_agents_enabled() -> bool:
    return os.environ.get("USE_REACT_AGENTS", "true").lower() in {"1", "true", "yes", "on"}


ATTRACTION_PROMPT = """你是一个景点筛选专家。根据用户的约束条件和知识库信息，使用高德地图工具搜索合适的景点。
用户约束：
- 城市：{city}
- 出行人群：{travelers}
- 偏好：{preferences}
- 天数：{days}天
{rag_context}

任务：
1. 搜索该城市的热门景点和特色景点
2. 结合知识库中的景点详情（适老适幼信息、开放时间、门票等）
3. 根据人群特点筛选：有老人需找适老景点，有小孩需找亲子景点
4. 返回景点列表，包含名称、地址、评分、类型、是否适老、是否亲子
请直接输出结构化结果。"""


async def attraction_node(state: AgentState) -> AgentState:
    constraints = state.get("constraints")
    if not constraints:
        return {
            "attractions": [],
            "traces": [_trace("attraction_agent", "缺少约束条件，跳过景点搜索")],
        }

    user_text = _first_user_text(state)
    query_parts = [
        constraints.city,
        user_text,
        "经典",
        "热门",
        "景点",
        "推荐",
        "亲子" if any(token in user_text for token in ("小孩", "孩子", "亲子")) else "",
        "历史文化" if any(token in user_text for token in ("历史", "文化")) else "",
        "慢节奏" if "慢" in user_text else "",
        "适老" if any(token in user_text for token in ("老人", "长辈", "父母")) else "",
        " ".join(constraints.preferences),
    ]
    retrieval_query = " ".join(part for part in query_parts if part).strip()

    rag_docs = []
    rag_context = ""
    try:
        retriever = TravelRetriever()
        rag_docs = retriever.retrieve_attractions(
            query=retrieval_query or f"{constraints.city} 景点推荐",
            city=constraints.city,
            k=4,
        )
        if rag_docs:
            rag_context = (
                "知识库参考信息："
                + chr(10)
                + chr(10).join(f"- {doc.page_content[:300]}" for doc in rag_docs)
            )
    except Exception as exc:
        rag_context = f"（知识库暂不可用: {exc}）"

    if not _react_agents_enabled():
        attractions = _attractions_from_docs(rag_docs, constraints.city)
        if not attractions:
            attractions = _fallback_attractions(constraints.city)
        return {
            "attractions": attractions,
            "sources": documents_to_sources(rag_docs),
            "traces": [
                _trace(
                    "attraction_agent",
                    f"快速景点检索完成，找到{len(attractions)}个景点；RAG命中{len(rag_docs)}条",
                    count=len(attractions),
                    evidence_count=len(rag_docs),
                    city=constraints.city,
                    rag_used=bool(rag_docs),
                )
            ],
        }

    try:
        llm = get_chat_model()
        from langgraph.prebuilt import create_react_agent

        prompt = ATTRACTION_PROMPT.format(
            city=constraints.city or "未知城市",
            travelers=", ".join(constraints.travelers) if constraints.travelers else "成人",
            preferences=", ".join(constraints.preferences)
            if constraints.preferences
            else "无特别偏好",
            days=constraints.days,
            rag_context=rag_context,
        )

        async with MCPToolRegistry().travel_tools() as tools:
            agent = create_react_agent(model=llm, tools=tools)
            response = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"recursion_limit": 8},
                ),
                timeout=agent_timeout_seconds(),
            )

        final_message = response["messages"][-1].content
        if isinstance(final_message, list):
            answer = "".join(
                item.get("text", "") for item in final_message if isinstance(item, dict)
            )
        else:
            answer = str(final_message)

        attractions = _parse_attractions(answer, constraints, rag_docs)
        return {
            "attractions": attractions,
            "messages": [AIMessage(content=answer)],
            "sources": documents_to_sources(rag_docs),
            "traces": [
                _trace(
                    "attraction_agent",
                    f"完成景点搜索，找到{len(attractions)}个景点；RAG命中{len(rag_docs)}条",
                    count=len(attractions),
                    evidence_count=len(rag_docs),
                    city=constraints.city,
                    rag_used=bool(rag_docs),
                )
            ],
        }
    except Exception as exc:
        attractions = _attractions_from_docs(rag_docs, constraints.city)
        if not attractions:
            attractions = _fallback_attractions(constraints.city)
        return {
            "attractions": attractions,
            "sources": documents_to_sources(rag_docs),
            "traces": [
                _trace(
                    "attraction_agent",
                    f"景点实时搜索失败，已使用{len(rag_docs)}条RAG证据生成兜底结果: {exc}",
                    error=str(exc),
                    evidence_count=len(rag_docs),
                    rag_used=bool(rag_docs),
                )
            ],
        }


def _parse_attractions(text: str, constraints, rag_docs=None) -> list[Attraction]:
    attractions: list[Attraction] = []
    try:
        fence = chr(96) * 3
        if fence + "json" in text:
            json_str = text.split(fence + "json", 1)[1].split(fence, 1)[0]
        elif fence in text:
            json_str = text.split(fence, 1)[1].split(fence, 1)[0]
        else:
            json_str = text

        data = json.loads(json_str)
        items = data if isinstance(data, list) else data.get("attractions", [])
        for item in items:
            attractions.append(
                Attraction(
                    name=item.get("name", ""),
                    address=item.get("address", ""),
                    rating=float(item.get("rating", 0)),
                    category=item.get("category", ""),
                    description=item.get("description", ""),
                    elderly_friendly=item.get("elderly_friendly", False),
                    child_friendly=item.get("child_friendly", False),
                )
            )
    except Exception:
        pass

    if not attractions:
        attractions = _attractions_from_docs(rag_docs, getattr(constraints, "city", ""))
    if not attractions:
        attractions = _fallback_attractions(constraints.city)
    return attractions


def _attractions_from_docs(rag_docs, city: str = "") -> list[Attraction]:
    attractions: list[Attraction] = []
    if not rag_docs:
        return attractions

    matched: list[Attraction] = []
    for doc in rag_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        name = str(metadata.get("name") or "").strip()
        if not name:
            continue
        item = Attraction(
            name=name,
            address=str(metadata.get("address") or "").strip(),
            rating=float(metadata.get("rating", 0.0) or 0.0),
            category=str(metadata.get("category") or "景点").strip() or "景点",
            description=str(metadata.get("summary") or doc.page_content[:180]).strip(),
            elderly_friendly=bool(metadata.get("elderly_friendly", False)),
            child_friendly=bool(metadata.get("child_friendly", False)),
            business_hours=str(metadata.get("business_hours", "") or ""),
            ticket_price=str(metadata.get("ticket_price", "") or ""),
        )
        attractions.append(item)
        doc_city = str(metadata.get("city", "") or "")
        if not city or city in doc_city or city in item.address:
            matched.append(item)
    return matched or attractions


def _fallback_attractions(city: str) -> list[Attraction]:
    label = city or "当地"
    return [
        Attraction(
            name=f"{label}经典景点",
            address="",
            rating=4.5,
            category="景点",
            child_friendly=True,
            elderly_friendly=True,
        ),
        Attraction(
            name=f"{label}特色文化点",
            address="",
            rating=4.3,
            category="景点",
            child_friendly=True,
            elderly_friendly=True,
        ),
        Attraction(
            name=f"{label}自然休闲点",
            address="",
            rating=4.2,
            category="景点",
            child_friendly=True,
            elderly_friendly=True,
        ),
    ]
