from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from gaode.agents.state import AgentState
from gaode.llm import get_chat_model
from gaode.schemas.travel import TraceEvent, TravelConstraints

KNOWN_CITIES = ("北京", "杭州", "成都", "西安")


def _trace(step: str, message: str, **data) -> TraceEvent:
    return TraceEvent(step=step, message=message, data=data)


def _first_user_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    message = next(
        (item for item in reversed(messages) if isinstance(item, HumanMessage)),
        messages[-1] if messages else HumanMessage(content=""),
    )
    content = message.content if isinstance(message, HumanMessage) else str(message)
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def _message_text(content) -> str:
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content or "")


def _extract_json_object(content) -> dict:
    text = _message_text(content).strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text.strip())
    if not isinstance(data, dict):
        raise ValueError("Supervisor response must be a JSON object")
    return data


def _fallback_constraints(user_text: str) -> TravelConstraints:
    city = ""
    for candidate in KNOWN_CITIES:
        if candidate in user_text:
            city = candidate
            break

    days_match = re.search(r"(\d+)\s*天", user_text)
    budget_match = re.search(r"(\d{3,5})\s*元?", user_text)

    travelers: list[str] = []
    if any(token in user_text for token in ("老人", "长辈", "父母")):
        travelers.append("老人")
    if any(token in user_text for token in ("小孩", "孩子", "亲子")):
        travelers.append("小孩")
    if not travelers:
        travelers.append("成人")

    dietary_restrictions: list[str] = []
    for token in ("不吃辣", "不要辣", "少辣", "清淡", "素食", "海鲜", "牛肉", "羊肉"):
        if token in user_text:
            dietary_restrictions.append(token)

    preferences: list[str] = []
    for token in ("亲子", "历史", "自然", "美食", "拍照", "慢节奏", "轻松"):
        if token in user_text:
            preferences.append(token)

    return TravelConstraints(
        city=city,
        days=int(days_match.group(1)) if days_match else 3,
        budget=float(budget_match.group(1)) if budget_match else 0,
        travelers=travelers,
        dietary_restrictions=dietary_restrictions,
        preferences=preferences,
        travel_style="均衡",
    )


def _city_from_text(user_text: str) -> str:
    for candidate in KNOWN_CITIES:
        if candidate in user_text:
            return candidate
    return ""


def _normalize_constraints(raw: dict, user_text: str) -> TravelConstraints:
    fallback = _fallback_constraints(user_text)
    city = str(raw.get("city", "") or "").strip()
    city = _city_from_text(user_text) or city
    if city and city not in KNOWN_CITIES:
        city = fallback.city

    try:
        days = int(raw.get("days", fallback.days) or fallback.days)
    except (TypeError, ValueError):
        days = fallback.days

    try:
        budget = float(raw.get("budget", fallback.budget) or fallback.budget)
    except (TypeError, ValueError):
        budget = fallback.budget

    travelers = raw.get("travelers", fallback.travelers)
    if not isinstance(travelers, list):
        travelers = fallback.travelers

    dietary_restrictions = raw.get("dietary_restrictions", fallback.dietary_restrictions)
    if not isinstance(dietary_restrictions, list):
        dietary_restrictions = fallback.dietary_restrictions

    preferences = raw.get("preferences", fallback.preferences)
    if not isinstance(preferences, list):
        preferences = fallback.preferences

    travel_style = str(
        raw.get("travel_style", fallback.travel_style) or fallback.travel_style
    ).strip()
    if travel_style not in {"慢节奏", "快节奏", "均衡"}:
        travel_style = fallback.travel_style

    return TravelConstraints(
        city=city or fallback.city,
        days=days if days > 0 else fallback.days,
        budget=budget if budget >= 0 else fallback.budget,
        travelers=[str(item) for item in travelers if str(item).strip()] or fallback.travelers,
        dietary_restrictions=[str(item) for item in dietary_restrictions if str(item).strip()],
        preferences=[str(item) for item in preferences if str(item).strip()],
        travel_style=travel_style,
    )


def _llm_supervisor_enabled() -> bool:
    return os.environ.get("USE_LLM_SUPERVISOR", "true").lower() in {"1", "true", "yes", "on"}


SUPERVISOR_PROMPT = """你是一个专业旅行规划系统的Supervisor。你的任务是分析用户需求，提取旅行约束条件。

请从用户输入中提取以下信息，以JSON格式返回：
{
    "city": "目的地城市",
    "days": 天数(数字),
    "budget": 预算(数字，元),
    "travelers": ["出行人群，如：老人、小孩、成人"],
    "dietary_restrictions": ["饮食忌口，如：辣、海鲜、素食"],
    "preferences": ["其他偏好，如：自然风光、历史文化、亲子"],
    "travel_style": "旅行风格：慢节奏/快节奏/均衡"
}

注意：
1. 如果用户没有明确说明某项，使用合理的默认值
2. 天数默认3天
3. 预算默认0表示不限
4. 旅行风格默认"均衡"
5. 只输出JSON，不要解释
"""


async def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor节点：解析用户需求，提取约束条件"""
    user_text = _first_user_text(state)

    if not _llm_supervisor_enabled():
        constraints = _fallback_constraints(user_text)
        return {
            "constraints": constraints,
            "traces": [
                _trace(
                    "supervisor",
                    "规则模式完成需求解析和约束提取",
                    constraints=constraints.model_dump(),
                    llm_used=False,
                )
            ],
        }

    try:
        llm = get_chat_model()
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=user_text),
            ]
        )
        constraints = _normalize_constraints(_extract_json_object(response.content), user_text)
        llm_used = True
        message = "LLM 完成需求解析和约束提取"
    except Exception as exc:
        constraints = _fallback_constraints(user_text)
        llm_used = False
        message = f"LLM 需求解析失败，已使用规则解析: {type(exc).__name__}"

    return {
        "constraints": constraints,
        "traces": [
            _trace(
                "supervisor",
                message,
                constraints=constraints.model_dump(),
                llm_used=llm_used,
            )
        ],
    }
