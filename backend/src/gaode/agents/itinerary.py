from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gaode.agents.state import AgentState
from gaode.llm import get_chat_model
from gaode.schemas.travel import GenerationMeta, TraceEvent, TravelPlan


def _trace(step: str, message: str, **data) -> TraceEvent:
    return TraceEvent(step=step, message=message, data=data)


ITINERARY_SYSTEM_PROMPT = "你是一个专业的旅行规划师。请输出简洁、可执行的行程，控制在 500 字以内。"


def _message_text(value) -> str:
    if isinstance(value, list):
        return "".join(item.get("text", "") for item in value if isinstance(item, dict))
    return str(value or "")


def _llm_itinerary_enabled() -> bool:
    return os.environ.get("USE_LLM_ITINERARY", "true").lower() in {"1", "true", "yes", "on"}


def _infer_rag_used(state: AgentState) -> tuple[bool, int]:
    traces = state.get("traces", [])
    evidence_count = 0
    rag_used = False
    for trace in traces:
        data = getattr(trace, "data", {}) or {}
        if data.get("rag_used"):
            rag_used = True
        count = data.get("count")
        if isinstance(count, int) and count > evidence_count:
            evidence_count = count
    return rag_used, evidence_count


async def _stream_markdown(
    stream_callback: Callable[[str], Awaitable[None]], markdown: str
) -> None:
    for line in markdown.splitlines(keepends=True):
        if line:
            await stream_callback(line)


def _build_prompt(state: AgentState) -> tuple[str, object, list, list]:
    constraints = state.get("constraints")
    attractions = state.get("attractions", [])
    routes = state.get("routes", [])
    dining_places = state.get("dining_places", [])

    if not constraints:
        return "", constraints, attractions, dining_places

    attractions_text = (
        chr(10).join(
            f"- {a.name}: {a.address} | {a.category} | 适老:{getattr(a, 'elderly_friendly', False)} | 亲子:{getattr(a, 'child_friendly', False)}"
            for a in attractions
        )
        if attractions
        else "暂无景点数据"
    )

    routes_text = (
        chr(10).join(
            f"- {r.origin} -> {r.destination}: {r.distance} / {r.duration} ({r.mode})"
            for r in routes
        )
        if routes
        else "暂无路线数据"
    )

    dining_text = (
        chr(10).join(
            f"- {d.name}: {d.address} | {d.category} | 人均{d.avg_price}元 | {', '.join(d.tags)}"
            for d in dining_places
        )
        if dining_places
        else "暂无餐厅数据"
    )

    prompt = (
        f"目的地：{constraints.city}"
        + chr(10)
        + f"天数：{constraints.days}天"
        + chr(10)
        + f"预算：{constraints.budget}元"
        + chr(10)
        + f"出行人群：{', '.join(constraints.travelers)}"
        + chr(10)
        + f"饮食忌口：{', '.join(constraints.dietary_restrictions) if constraints.dietary_restrictions else '无'}"
        + chr(10)
        + f"旅行风格：{constraints.travel_style}"
        + chr(10)
        + chr(10)
        + "已筛选的景点："
        + chr(10)
        + attractions_text
        + chr(10)
        + chr(10)
        + "已规划的路线："
        + chr(10)
        + routes_text
        + chr(10)
        + chr(10)
        + "推荐的餐厅："
        + chr(10)
        + dining_text
        + chr(10)
        + chr(10)
        + "任务："
        + chr(10)
        + "1. 将景点分配到每天"
        + chr(10)
        + "2. 安排每天上午、下午、晚上"
        + chr(10)
        + "3. 安排每日餐饮"
        + chr(10)
        + "4. 考虑路线衔接和移动时间"
        + chr(10)
        + "5. 基于以上知识库信息生成简洁 Markdown 行程，每天 3-5 条要点，总字数尽量控制在 500 字以内"
        + chr(10)
    )
    return prompt, constraints, attractions, dining_places


async def generate_itinerary(
    state: AgentState,
    stream_callback: Callable[[str], Awaitable[None]] | None = None,
) -> AgentState:
    prompt, constraints, attractions, dining_places = _build_prompt(state)
    if not constraints:
        return {
            "answer": "缺少约束条件，无法生成行程。",
            "generation_meta": GenerationMeta(
                mode="fallback",
                rag_used=False,
                llm_used=False,
                evidence_count=0,
                summary="缺少约束条件",
            ),
            "traces": [_trace("itinerary_agent", "缺少约束条件")],
        }

    rag_used, evidence_count = _infer_rag_used(state)

    if not _llm_itinerary_enabled():
        markdown = _fallback_markdown(
            constraints.city, constraints.days, attractions, dining_places
        )
        if stream_callback is not None:
            await _stream_markdown(stream_callback, markdown)

        plan = TravelPlan(
            city=constraints.city,
            days=constraints.days,
            constraints=constraints,
            markdown=markdown,
            tips=["建议提前预订酒店", "景点门票可提前购买", "注意天气变化"],
        )

        return {
            "plan": plan,
            "answer": markdown,
            "messages": [AIMessage(content=markdown)],
            "generation_meta": GenerationMeta(
                mode="rag_only" if rag_used else "fallback",
                rag_used=rag_used,
                llm_used=False,
                evidence_count=evidence_count,
                summary="本地模板生成" if not rag_used else "检索信息驱动的模板生成",
            ),
            "traces": [
                _trace(
                    "itinerary_agent",
                    f"快速行程模板生成完成，输出{constraints.days}天{constraints.city}行程",
                    city=constraints.city,
                    days=constraints.days,
                    fast_mode=True,
                )
            ],
        }

    try:
        llm = get_chat_model()
        messages = [
            SystemMessage(content=ITINERARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        if stream_callback is None:
            response = await llm.ainvoke(messages)
            markdown = _message_text(response.content).strip()
        else:
            chunks: list[str] = []
            async for chunk in llm.astream(messages):
                text = _message_text(getattr(chunk, "content", ""))
                if text:
                    chunks.append(text)
                    await stream_callback(text)
            markdown = "".join(chunks).strip()

        if not markdown:
            markdown = _fallback_markdown(
                constraints.city, constraints.days, attractions, dining_places
            )

        plan = TravelPlan(
            city=constraints.city,
            days=constraints.days,
            constraints=constraints,
            markdown=markdown,
            tips=["建议提前预订酒店", "景点门票可提前购买", "注意天气变化"],
        )

        return {
            "plan": plan,
            "answer": markdown,
            "messages": [AIMessage(content=markdown)],
            "generation_meta": GenerationMeta(
                mode="llm_rag" if rag_used else "llm_only",
                rag_used=rag_used,
                llm_used=True,
                evidence_count=evidence_count,
                summary="LLM 已基于知识库信息生成行程" if rag_used else "LLM 未检索到知识库证据",
            ),
            "traces": [
                _trace(
                    "itinerary_agent",
                    f"完成{constraints.days}天{constraints.city}行程生成",
                    city=constraints.city,
                    days=constraints.days,
                )
            ],
        }
    except Exception as exc:
        fallback_markdown = _fallback_markdown(
            constraints.city, constraints.days, attractions, dining_places
        )
        fallback_plan = TravelPlan(
            city=constraints.city,
            days=constraints.days,
            constraints=constraints,
            markdown=fallback_markdown,
            tips=[
                "当前实时服务暂不可用，景点和餐厅信息为本地知识库兜底结果",
                "出行前请确认开放时间、门票和交通情况",
            ],
        )
        return {
            "plan": fallback_plan,
            "answer": fallback_markdown,
            "generation_meta": GenerationMeta(
                mode="fallback",
                rag_used=rag_used,
                llm_used=False,
                evidence_count=evidence_count,
                summary=f"LLM 生成失败: {type(exc).__name__}",
            ),
            "traces": [
                _trace(
                    "itinerary_agent",
                    f"实时行程生成失败，已返回兜底计划: {type(exc).__name__}",
                    error=str(exc) or type(exc).__name__,
                )
            ],
        }


async def itinerary_node(state: AgentState) -> AgentState:
    return await generate_itinerary(state)


def _fallback_markdown(city: str, days: int, attractions, dining_places) -> str:
    attraction_names = [item.name for item in attractions if getattr(item, "name", "")] or [
        "当地热门景点"
    ]
    dining_names = [item.name for item in dining_places if getattr(item, "name", "")] or [
        "当地特色餐厅"
    ]

    lines = [f"## {city}{days}天行程", ""]
    for day in range(1, days + 1):
        attraction = attraction_names[(day - 1) % len(attraction_names)]
        dining = dining_names[(day - 1) % len(dining_names)]
        lines.extend(
            [
                f"### Day {day}",
                f"- 上午：游览 {attraction}",
                "- 午餐：根据预算和忌口选择附近餐厅",
                f"- 下午：继续游览 {attraction}",
                f"- 晚餐：{dining}",
                "",
            ]
        )

    lines.extend(
        [
            "## 旅行贴士",
            "- 出发前确认天气、开放时间、门票和交通情况",
            "- 老人和儿童同行时，建议减少连续步行并预留休息时间",
        ]
    )
    return chr(10).join(lines)
