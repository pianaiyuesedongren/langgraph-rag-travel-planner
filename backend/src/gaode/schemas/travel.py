from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TravelRequest(BaseModel):
    """旅行规划请求"""

    question: str = Field(..., min_length=1, max_length=4000, description="用户需求描述")
    thread_id: str | None = Field(default=None, max_length=128)


class TravelConstraints(BaseModel):
    """旅行约束条件"""

    city: str = Field(default="", description="目的地城市")
    days: int = Field(default=3, description="旅行天数")
    budget: float = Field(default=0, description="预算(元)")
    travelers: list[str] = Field(default_factory=list, description="出行人群: 老人/小孩/成人等")
    dietary_restrictions: list[str] = Field(default_factory=list, description="饮食忌口")
    preferences: list[str] = Field(default_factory=list, description="其他偏好")
    travel_style: str = Field(default="balanced", description="旅行风格: 慢节奏/快节奏/均衡")


class Attraction(BaseModel):
    """景点信息"""

    name: str
    address: str = ""
    rating: float = 0.0
    category: str = ""
    description: str = ""
    elderly_friendly: bool = False
    child_friendly: bool = False
    latitude: float = 0.0
    longitude: float = 0.0
    business_hours: str = ""
    ticket_price: str = ""


class RouteInfo(BaseModel):
    """路线信息"""

    origin: str
    destination: str
    distance: str = ""
    duration: str = ""
    mode: str = "driving"  # driving / transit
    steps: list[str] = Field(default_factory=list)


class DiningPlace(BaseModel):
    """餐饮地点"""

    name: str
    address: str = ""
    rating: float = 0.0
    category: str = ""
    avg_price: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    tags: list[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    """单日行程"""

    day: int
    date: str = ""
    attractions: list[Attraction] = Field(default_factory=list)
    dining: list[DiningPlace] = Field(default_factory=list)
    routes: list[RouteInfo] = Field(default_factory=list)
    summary: str = ""


class TravelPlan(BaseModel):
    """完整旅行计划"""

    city: str
    days: int
    constraints: TravelConstraints
    daily_plans: list[DayPlan] = Field(default_factory=list)
    total_budget_estimate: str = ""
    tips: list[str] = Field(default_factory=list)
    markdown: str = ""


class GenerationMeta(BaseModel):
    """生成来源与工作模式"""

    mode: str = Field(default="fallback")
    rag_used: bool = Field(default=False)
    llm_used: bool = Field(default=False)
    evidence_count: int = Field(default=0)
    summary: str = Field(default="")


class EvidenceSource(BaseModel):
    """Knowledge item used to ground the generated plan."""

    source: str
    url: str = ""
    title: str = ""
    kind: str = "general"
    city: str = ""
    score: float = 0.0


class TraceEvent(BaseModel):
    """追踪事件"""

    step: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TravelResponse(BaseModel):
    """旅行规划响应"""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str
    plan: TravelPlan | None = None
    answer: str = ""
    traces: list[TraceEvent] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)
    generation_meta: GenerationMeta = Field(default_factory=GenerationMeta)
