from __future__ import annotations

import asyncio
import json
import os

from langchain_core.messages import AIMessage, HumanMessage

from gaode.agents.state import AgentState, agent_timeout_seconds
from gaode.llm import get_chat_model
from gaode.rag.retriever import TravelRetriever, documents_to_sources
from gaode.schemas.travel import DiningPlace, TraceEvent
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


DINING_PROMPT = """你是一个餐饮推荐专家。根据用户的约束条件、当前景点和知识库信息，使用高德地图工具搜索合适的餐厅。
景点列表：
{attractions}

用户约束：
- 城市：{city}
- 饮食忌口：{dietary_restrictions}
- 预算：{budget}元
- 出行人群：{travelers}

{rag_context}

任务：
1. 使用高德地图搜索景点周边的餐厅
2. 结合知识库中的餐厅详情（口味、评价、特色菜等）
3. 根据饮食忌口过滤
4. 考虑老人和小孩的口味需求
5. 返回餐厅列表，包含名称、地址、评分、类型、人均价格、标签
请直接输出结构化结果。"""


async def dining_node(state: AgentState) -> AgentState:
    constraints = state.get("constraints")
    attractions = state.get("attractions", [])

    if not constraints:
        return {
            "dining_places": [],
            "traces": [_trace("dining_agent", "缺少约束条件，跳过餐饮推荐")],
        }

    attractions_text = (
        chr(10).join(f"- {a.name}: {a.address}" for a in attractions)
        if attractions
        else "暂无景点信息，请推荐当地特色餐厅"
    )
    user_text = _first_user_text(state)
    query_parts = [
        constraints.city,
        user_text,
        "餐厅",
        "美食",
        "特色",
        "老字号",
        "推荐",
        " ".join(constraints.dietary_restrictions),
        " ".join(constraints.preferences),
    ]
    retrieval_query = " ".join(part for part in query_parts if part).strip()

    rag_docs = []
    rag_context = ""
    try:
        retriever = TravelRetriever()
        rag_docs = retriever.retrieve_restaurants(
            query=retrieval_query
            or f"{constraints.city} 餐厅推荐 {' '.join(constraints.dietary_restrictions)}",
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
        dining_places = _dining_from_docs(rag_docs, constraints.city)
        if not dining_places:
            dining_places = _fallback_dining(constraints.city)
        return {
            "dining_places": dining_places,
            "sources": documents_to_sources(rag_docs),
            "traces": [
                _trace(
                    "dining_agent",
                    f"快速餐饮推荐完成，共{len(dining_places)}家餐厅",
                    count=len(dining_places),
                    rag_used=bool(rag_docs),
                )
            ],
        }

    try:
        llm = get_chat_model()
        from langgraph.prebuilt import create_react_agent

        prompt = DINING_PROMPT.format(
            attractions=attractions_text,
            city=constraints.city or "未知城市",
            dietary_restrictions=", ".join(constraints.dietary_restrictions)
            if constraints.dietary_restrictions
            else "无",
            budget=constraints.budget,
            travelers=", ".join(constraints.travelers) if constraints.travelers else "成人",
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

        dining_places = _parse_dining(answer, rag_docs, constraints.city)
        return {
            "dining_places": dining_places,
            "messages": [AIMessage(content=answer)],
            "sources": documents_to_sources(rag_docs),
            "traces": [
                _trace(
                    "dining_agent",
                    f"完成餐饮推荐，共{len(dining_places)}家餐厅",
                    count=len(dining_places),
                    rag_used=bool(rag_docs),
                )
            ],
        }
    except Exception as exc:
        dining_places = _dining_from_docs(rag_docs, constraints.city)
        if not dining_places:
            dining_places = _fallback_dining(constraints.city)
        return {
            "dining_places": dining_places,
            "sources": documents_to_sources(rag_docs),
            "traces": [
                _trace(
                    "dining_agent",
                    f"餐饮推荐失败，已返回兜底结果: {exc}",
                    error=str(exc),
                )
            ],
        }


def _parse_dining(text: str, rag_docs=None, city: str = "") -> list[DiningPlace]:
    dining_places: list[DiningPlace] = []
    try:
        fence = chr(96) * 3
        if fence + "json" in text:
            json_str = text.split(fence + "json", 1)[1].split(fence, 1)[0]
        elif fence in text:
            json_str = text.split(fence, 1)[1].split(fence, 1)[0]
        else:
            json_str = text

        data = json.loads(json_str)
        items = (
            data
            if isinstance(data, list)
            else data.get("restaurants", data.get("dining_places", []))
        )
        for item in items:
            dining_places.append(
                DiningPlace(
                    name=item.get("name", ""),
                    address=item.get("address", ""),
                    rating=float(item.get("rating", 0)),
                    category=item.get("category", ""),
                    avg_price=float(item.get("avg_price", 0)),
                    tags=item.get("tags", []),
                )
            )
    except Exception:
        pass

    if not dining_places:
        dining_places = _dining_from_docs(rag_docs, city)
        if not dining_places:
            dining_places = _fallback_dining("当地")
    return dining_places


def _dining_from_docs(rag_docs, city: str = "") -> list[DiningPlace]:
    dining_places: list[DiningPlace] = []
    if not rag_docs:
        return dining_places

    matched: list[DiningPlace] = []
    for doc in rag_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        name = str(metadata.get("name") or "").strip()
        if not name:
            continue
        tags = metadata.get("tags")
        if not isinstance(tags, list):
            tags = []
        item = DiningPlace(
            name=name,
            address=str(metadata.get("address") or "").strip(),
            rating=float(metadata.get("rating", 0.0) or 0.0),
            category=str(metadata.get("category") or "餐厅").strip() or "餐厅",
            avg_price=float(metadata.get("avg_price", 0.0) or 0.0),
            tags=[str(tag) for tag in tags if str(tag).strip()],
        )
        dining_places.append(item)
        doc_city = str(metadata.get("city", "") or "")
        if not city or city in doc_city or city in item.address:
            matched.append(item)
    return matched or dining_places


def _fallback_dining(city: str) -> list[DiningPlace]:
    label = city or "当地"
    return [
        DiningPlace(
            name=f"{label}特色餐厅",
            address="",
            rating=4.5,
            category="餐厅",
            avg_price=80,
            tags=["特色", "本地风味"],
        ),
        DiningPlace(
            name=f"{label}家常菜馆",
            address="",
            rating=4.3,
            category="餐厅",
            avg_price=60,
            tags=["家常菜", "平价"],
        ),
        DiningPlace(
            name=f"{label}小吃推荐",
            address="",
            rating=4.2,
            category="餐厅",
            avg_price=40,
            tags=["小吃", "便捷"],
        ),
    ]
