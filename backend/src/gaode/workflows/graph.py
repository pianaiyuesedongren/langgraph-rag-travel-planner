from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from gaode.agents.attraction import attraction_node
from gaode.agents.dining import dining_node
from gaode.agents.itinerary import generate_itinerary, itinerary_node
from gaode.agents.route import route_node
from gaode.agents.state import AgentState
from gaode.agents.supervisor import supervisor_node
from gaode.config import get_settings
from gaode.infra.redis import get_redis_manager
from gaode.memory.checkpoint import get_memory_manager, get_memory_saver
from gaode.schemas.travel import TravelResponse

PLAN_CACHE_VERSION = "v2"


def build_graph(include_itinerary: bool = False, checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("attraction_agent", attraction_node)
    builder.add_node("route_agent", route_node)
    builder.add_node("dining_agent", dining_node)

    if include_itinerary:
        builder.add_node("itinerary_agent", itinerary_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "attraction_agent")
    builder.add_edge("attraction_agent", "route_agent")
    builder.add_edge("attraction_agent", "dining_agent")

    if include_itinerary:
        builder.add_edge("route_agent", "itinerary_agent")
        builder.add_edge("dining_agent", "itinerary_agent")
        builder.add_edge("itinerary_agent", END)
    else:
        builder.add_edge("route_agent", END)
        builder.add_edge("dining_agent", END)

    return builder.compile(checkpointer=checkpointer)


def run_travel_planning(
    question: str,
    thread_id: str | None = None,
    use_memory: bool = True,
) -> TravelResponse:
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question cannot be empty")

    return asyncio.run(
        arun_travel_planning(clean_question, thread_id=thread_id, use_memory=use_memory)
    )


async def arun_travel_planning(
    question: str,
    thread_id: str | None = None,
    use_memory: bool = True,
) -> TravelResponse:
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question cannot be empty")

    final_thread_id = thread_id or str(uuid4())
    redis = get_redis_manager()
    cache_key = f"travel:{PLAN_CACHE_VERSION}:plan:{redis.request_key(clean_question, thread_id, use_memory)}"

    cached = await redis.get_json(cache_key)
    if cached:
        cached_response = TravelResponse.model_validate(cached)
        return cached_response.model_copy(
            update={"request_id": str(uuid4()), "thread_id": final_thread_id}
        )

    if use_memory:
        checkpointer = await get_memory_saver()
        graph = build_graph(include_itinerary=False, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": final_thread_id}}
        memory_mgr = get_memory_manager()
        await memory_mgr.add_message(final_thread_id, "user", clean_question)
    else:
        graph = build_graph(include_itinerary=False)
        config = {}
        memory_mgr = None

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=clean_question)]}, config=config
    )

    core_traces = list(result.get("traces", []))
    final_result = await generate_itinerary(result)
    answer = str(final_result.get("answer", ""))
    if memory_mgr:
        await memory_mgr.add_message(final_thread_id, "assistant", answer)

    response = TravelResponse(
        request_id=str(uuid4()),
        thread_id=final_thread_id,
        plan=final_result.get("plan"),
        answer=answer,
        traces=core_traces + list(final_result.get("traces", [])),
        sources=result.get("sources", []),
        generation_meta=final_result.get("generation_meta"),
    )
    if get_settings().database_enabled:
        from gaode.db.repository import save_travel_response

        await save_travel_response(clean_question, response)
    await redis.set_json(cache_key, response.model_dump(mode="json"))
    return response


async def stream_travel_planning(
    question: str,
    thread_id: str | None = None,
    use_memory: bool = True,
    event_job_id: str | None = None,
):
    clean_question = question.strip()
    if not clean_question:
        yield {
            "event": "error",
            "data": json.dumps({"error": "question cannot be empty"}, ensure_ascii=False),
        }
        return

    final_thread_id = thread_id or str(uuid4())
    redis = get_redis_manager()
    event_key = event_job_id or final_thread_id
    cache_key = f"travel:{PLAN_CACHE_VERSION}:plan:{redis.request_key(clean_question, thread_id, use_memory)}"

    async def emit(event: dict):
        await redis.publish_event(event_key, event)
        return event

    yield await emit(
        {"event": "start", "data": json.dumps({"thread_id": final_thread_id}, ensure_ascii=False)}
    )

    cached = await redis.get_json(cache_key)
    if cached:
        cached_response = TravelResponse.model_validate(cached).model_copy(
            update={"request_id": str(uuid4()), "thread_id": final_thread_id}
        )
        yield await emit(
            {
                "event": "progress",
                "data": json.dumps(
                    {"message": "命中 Redis 缓存，正在返回结果..."}, ensure_ascii=False
                ),
            }
        )
        for offset in range(0, len(cached_response.answer), 180):
            yield await emit(
                {"event": "token", "data": cached_response.answer[offset : offset + 180]}
            )
        yield await emit({"event": "complete", "data": cached_response.model_dump_json()})
        return

    memory_mgr = get_memory_manager() if use_memory else None
    if memory_mgr:
        await memory_mgr.add_message(final_thread_id, "user", clean_question)

    if use_memory:
        graph = build_graph(include_itinerary=False, checkpointer=await get_memory_saver())
        config = {"configurable": {"thread_id": final_thread_id}}
    else:
        graph = build_graph(include_itinerary=False)
        config = {}

    result_state: dict = {}
    collected_traces: list = []
    started_at = asyncio.get_running_loop().time()
    try:
        async for graph_event in graph.astream_events(
            {"messages": [HumanMessage(content=clean_question)]},
            config=config,
            version="v2",
        ):
            event_name = graph_event.get("event", "")
            node_name = graph_event.get("name", "")
            if event_name == "on_chain_start" and node_name in {
                "supervisor",
                "attraction_agent",
                "route_agent",
                "dining_agent",
            }:
                yield await emit(
                    {
                        "event": "progress",
                        "data": json.dumps(
                            {"step": node_name, "message": f"{node_name} 执行中..."},
                            ensure_ascii=False,
                        ),
                    }
                )
            elif event_name == "on_chain_end" and node_name in {
                "supervisor",
                "attraction_agent",
                "route_agent",
                "dining_agent",
            }:
                output = graph_event.get("data", {}).get("output")
                if isinstance(output, dict):
                    traces = output.get("traces")
                    if isinstance(traces, list):
                        collected_traces.extend(traces)
                    result_state.update(output)

        async def forward_token(token: str):
            await emit({"event": "token", "data": token})

        yield await emit(
            {
                "event": "progress",
                "data": json.dumps({"message": "开始生成行程..."}, ensure_ascii=False),
            }
        )
        final_result = await generate_itinerary(result_state, stream_callback=forward_token)

        answer = str(final_result.get("answer", ""))
        if memory_mgr:
            await memory_mgr.add_message(final_thread_id, "assistant", answer)

        response = TravelResponse(
            request_id=str(uuid4()),
            thread_id=final_thread_id,
            plan=final_result.get("plan"),
            answer=answer,
            traces=collected_traces + list(final_result.get("traces", [])),
            sources=result_state.get("sources", []),
            generation_meta=final_result.get("generation_meta"),
        )
        if get_settings().database_enabled:
            from gaode.db.repository import save_travel_response

            await save_travel_response(clean_question, response)
        await redis.set_json(cache_key, response.model_dump(mode="json"))
        elapsed = asyncio.get_running_loop().time() - started_at
        yield await emit(
            {
                "event": "progress",
                "data": json.dumps(
                    {"message": f"规划完成，用时{elapsed:.1f}秒"}, ensure_ascii=False
                ),
            }
        )
        yield await emit({"event": "complete", "data": response.model_dump_json()})
    except Exception as exc:
        yield await emit(
            {
                "event": "error",
                "data": json.dumps({"error": str(exc) or type(exc).__name__}, ensure_ascii=False),
            }
        )
