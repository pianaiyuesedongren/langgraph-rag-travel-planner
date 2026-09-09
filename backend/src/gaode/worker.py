"""Redis-backed planning worker.

Run with:
    python -m gaode.worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

for key, value in {
    "GRPC_VERBOSITY": "ERROR",
    "GRPC_CPP_MIN_LOG_LEVEL": "2",
    "GLOG_minloglevel": "2",
}.items():
    os.environ.setdefault(key, value)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaode.infra.redis import get_redis_manager
from gaode.workflows.graph import stream_travel_planning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gaode.worker")


async def run_worker() -> None:
    redis = get_redis_manager()
    logger.info("Travel planning worker started")
    while True:
        job = await redis.dequeue_job(timeout=5)
        if not job:
            continue

        job_id = job["job_id"]
        logger.info("Processing planning job %s", job_id)
        try:
            async for _ in stream_travel_planning(
                job["question"],
                # Keep a missing thread_id as None so identical stateless jobs
                # share the Redis cache; event_job_id remains the queue id.
                thread_id=job.get("thread_id"),
                use_memory=job.get("use_memory", True),
                event_job_id=job_id,
            ):
                pass
        except Exception as exc:
            logger.exception("Planning job %s failed", job_id)
            await redis.publish_event(
                job_id,
                {"event": "error", "data": str(exc)},
            )


if __name__ == "__main__":
    asyncio.run(run_worker())
