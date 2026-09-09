from __future__ import annotations

import asyncio
import os

from langchain_core.messages import AIMessage

from gaode.agents.state import AgentState, agent_timeout_seconds
from gaode.llm import get_chat_model
from gaode.schemas.travel import RouteInfo, TraceEvent
from gaode.tools.mcp_registry import MCPToolRegistry


def _trace(step: str, message: str, **data) -> TraceEvent:
    return TraceEvent(step=step, message=message, data=data)


def _react_agents_enabled() -> bool:
    return os.environ.get("USE_REACT_AGENTS", "true").lower() in {"1", "true", "yes", "on"}


def _routes_from_attractions(attractions) -> list[RouteInfo]:
    routes: list[RouteInfo] = []
    if len(attractions) > 1:
        for i in range(len(attractions) - 1):
            routes.append(
                RouteInfo(
                    origin=attractions[i].name,
                    destination=attractions[i + 1].name,
                    distance="约3公里",
                    duration="约10分钟",
                    mode="driving",
                )
            )
    return routes


ROUTE_PROMPT = """你是一个路线优化专家。根据已筛选的景点列表，使用高德地图工具规划最优路线。

景点列表：
{attractions}

用户约束：
- 天数：{days}天
- 预算：{budget}元
- 旅行风格：{travel_style}
- 出行人群：{travelers}

任务：
1. 使用高德地图工具获取景点间的驾车/公交路线
2. 按天数合理分配景点
3. 考虑时间效率和体力消耗
4. 输出每日行程路线，包含出发地、目的地、距离、时长、交通方式

请使用路线规划工具获取实际路线数据。
"""


async def route_node(state: AgentState) -> AgentState:
    """路线优化Agent：规划最优游览路线"""
    constraints = state.get("constraints")
    attractions = state.get("attractions", [])

    if not constraints or not attractions:
        return {
            "routes": [],
            "traces": [_trace("route_agent", "缺少景点或约束，跳过路线规划")],
        }

    attractions_text = "\n".join([f"- {a.name}: {a.address} ({a.category})" for a in attractions])

    if not _react_agents_enabled():
        routes = _routes_from_attractions(attractions)
        return {
            "routes": routes,
            "traces": [
                _trace(
                    "route_agent",
                    f"快速路线规划完成，共{len(routes)}条路线",
                    count=len(routes),
                )
            ],
        }

    try:
        llm = get_chat_model()
        from langgraph.prebuilt import create_react_agent

        prompt = ROUTE_PROMPT.format(
            attractions=attractions_text,
            days=constraints.days,
            budget=constraints.budget,
            travel_style=constraints.travel_style,
            travelers=", ".join(constraints.travelers),
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

        routes = _parse_routes(answer, attractions)

        return {
            "routes": routes,
            "messages": [AIMessage(content=answer)],
            "traces": [
                _trace(
                    "route_agent",
                    f"完成路线规划，共{len(routes)}条路线",
                    count=len(routes),
                )
            ],
        }

    except Exception as exc:
        return {
            "routes": [],
            "traces": [
                _trace(
                    "route_agent",
                    f"路线规划失败: {exc}",
                    error=str(exc),
                )
            ],
        }


def _parse_routes(text: str, attractions) -> list[RouteInfo]:
    """解析LLM返回的路线信息"""
    routes = []
    try:
        import json

        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0]
        else:
            json_str = text

        data = json.loads(json_str)
        items = data if isinstance(data, list) else data.get("routes", [])
        for item in items:
            routes.append(
                RouteInfo(
                    origin=item.get("origin", ""),
                    destination=item.get("destination", ""),
                    distance=item.get("distance", ""),
                    duration=item.get("duration", ""),
                    mode=item.get("mode", "driving"),
                    steps=item.get("steps", []),
                )
            )
    except Exception:
        pass

    if not routes and len(attractions) > 1:
        for i in range(len(attractions) - 1):
            routes.append(
                RouteInfo(
                    origin=attractions[i].name,
                    destination=attractions[i + 1].name,
                    distance="约3公里",
                    duration="约10分钟",
                    mode="driving",
                )
            )

    return routes
