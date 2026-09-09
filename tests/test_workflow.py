import json

import pytest
from gaode.agents.itinerary import _infer_rag_used
from gaode.schemas.travel import (
    EvidenceSource,
    GenerationMeta,
    TraceEvent,
    TravelConstraints,
    TravelPlan,
)
from gaode.workflows import graph as workflow


def test_rag_status_comes_from_evidence_not_last_agent_trace() -> None:
    state = {
        "sources": [
            EvidenceSource(
                source="resource/knowledge/attractions/故宫博物院.md",
                title="故宫博物院",
                kind="attraction",
                city="北京",
            )
        ],
        "traces": [TraceEvent(step="route_agent", message="共0条路线")],
    }

    assert _infer_rag_used(state) == (True, 1)


class FakeGraph:
    async def astream_events(self, *_args, **_kwargs):
        yield {
            "event": "on_chain_end",
            "name": "route_agent",
            "data": {
                "output": {"routes": [], "traces": [TraceEvent(step="route", message="done")]}
            },
        }
        source = EvidenceSource(
            source="resource/knowledge/attractions/故宫博物院.md",
            title="故宫博物院",
            kind="attraction",
            city="北京",
        )
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "constraints": TravelConstraints(city="北京", days=2),
                    "sources": [source],
                    "traces": [
                        TraceEvent(
                            step="attraction_agent",
                            message="RAG命中1条",
                            data={"rag_used": True},
                        )
                    ],
                }
            },
        }


@pytest.mark.asyncio
async def test_stream_uses_reducer_merged_root_state(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "build_graph", lambda **_kwargs: FakeGraph())

    async def fake_generate(state, stream_callback=None):
        assert len(state["sources"]) == 1
        if stream_callback:
            await stream_callback("done")
        return {
            "plan": TravelPlan(
                city="北京",
                days=2,
                constraints=state["constraints"],
                markdown="done",
            ),
            "answer": "done",
            "traces": [],
            "generation_meta": GenerationMeta(
                mode="llm_rag",
                rag_used=True,
                llm_used=True,
                evidence_count=1,
            ),
        }

    monkeypatch.setattr(workflow, "generate_itinerary", fake_generate)
    events = [
        event async for event in workflow.stream_travel_planning("2天北京游", use_memory=False)
    ]
    completed = next(event for event in events if event["event"] == "complete")
    payload = json.loads(completed["data"])
    assert payload["generation_meta"]["mode"] == "llm_rag"
    assert [source["title"] for source in payload["sources"]] == ["故宫博物院"]
