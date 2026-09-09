from __future__ import annotations

import os
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage

from gaode.schemas.travel import (
    Attraction,
    DayPlan,
    DiningPlace,
    EvidenceSource,
    RouteInfo,
    TraceEvent,
    TravelConstraints,
    TravelPlan,
)


class AgentState(TypedDict, total=False):
    """多智能体共享状态"""

    messages: Annotated[list[AnyMessage], add]
    user_query: str
    constraints: TravelConstraints
    attractions: list[Attraction]
    routes: list[RouteInfo]
    dining_places: list[DiningPlace]
    daily_plans: list[DayPlan]
    plan: TravelPlan
    answer: str
    traces: Annotated[list[TraceEvent], add]
    sources: Annotated[list[EvidenceSource], add]


def agent_timeout_seconds() -> float:
    """Maximum time allowed for one external-tool Agent invocation."""
    try:
        return max(12.0, float(os.environ.get("AGENT_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120.0
