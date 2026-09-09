from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

for key, value in {
    "GRPC_VERBOSITY": "ERROR",
    "GRPC_CPP_MIN_LOG_LEVEL": "2",
    "GLOG_minloglevel": "2",
}.items():
    os.environ.setdefault(key, value)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

BACKEND_SRC = Path(__file__).resolve().parent / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from gaode.db import close_database, database_health, init_database
from gaode.infra.redis import get_redis_manager
from gaode.memory.checkpoint import close_memory, get_memory_manager, init_memory
from gaode.schemas.travel import TravelResponse
from gaode.workflows.graph import arun_travel_planning, stream_travel_planning


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_database()
    await init_memory()
    try:
        yield
    finally:
        await get_redis_manager().close()
        await close_memory()
        await close_database()


app = FastAPI(title="Gaode Travel Planner", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlanRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None
    use_memory: bool = True


class JobResponse(BaseModel):
    job_id: str
    status: str
    events_url: str


@app.post("/api/plan/jobs", response_model=JobResponse, status_code=202)
async def enqueue_plan(request: PlanRequest):
    """Queue a durable planning job for the separate Redis worker."""
    job_id = str(uuid4())
    redis = get_redis_manager()
    queued = await redis.enqueue_job(
        {
            "job_id": job_id,
            "question": request.question,
            "thread_id": request.thread_id,
            "use_memory": request.use_memory,
        }
    )
    if not queued:
        raise HTTPException(status_code=503, detail="Redis job queue unavailable")

    await redis.publish_event(
        job_id,
        {"event": "queued", "data": json.dumps({"job_id": job_id})},
    )
    return JobResponse(
        job_id=job_id,
        status="queued",
        events_url=f"/api/plan/stream/{job_id}",
    )


@app.post("/api/plan")
async def create_plan(request: PlanRequest) -> TravelResponse:
    response = await arun_travel_planning(
        request.question,
        request.thread_id,
        use_memory=request.use_memory,
    )
    return response


@app.post("/api/plan/stream")
async def create_plan_stream(request: PlanRequest):
    async def event_generator():
        async for event in stream_travel_planning(
            request.question,
            request.thread_id,
            use_memory=request.use_memory,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str):
    memory_mgr = get_memory_manager()
    history = await memory_mgr.get_history(thread_id)
    return {"thread_id": thread_id, "messages": history}


@app.delete("/api/history/{thread_id}")
async def clear_history(thread_id: str):
    memory_mgr = get_memory_manager()
    await memory_mgr.clear_history(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


@app.get("/api/health")
async def health_check():
    redis_health = await get_redis_manager().health()
    postgres_health = await database_health()
    return {
        "status": "ok",
        "service": "Gaode Travel Planner",
        "version": "2.1.0",
        "redis": redis_health,
        "postgres": postgres_health,
    }


@app.get("/api/plan/stream/{job_id}")
async def replay_plan_events(job_id: str, last_id: str = "0-0"):
    """Replay persisted Redis Stream events after a client reconnects."""
    redis = get_redis_manager()

    async def event_generator():
        cursor = last_id
        idle_rounds = 0
        while idle_rounds < 4:
            events = await redis.read_events(job_id, cursor, block_ms=15000)
            if not events:
                idle_rounds += 1
                continue
            idle_rounds = 0
            for event_id, event in events:
                cursor = event_id
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("event") in {"complete", "error"}:
                    return

        yield f"data: {json.dumps({'event': 'error', 'data': json.dumps({'error': 'event stream expired'})}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
