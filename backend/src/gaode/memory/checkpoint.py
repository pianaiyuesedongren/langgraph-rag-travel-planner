from __future__ import annotations

import logging
from datetime import UTC, datetime

from langgraph.checkpoint.memory import MemorySaver

from gaode.config import get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Conversation history facade with PostgreSQL and local-dev fallback."""

    def __init__(self):
        self._conversations: dict[str, list[dict]] = {}

    async def get_history(self, thread_id: str) -> list[dict]:
        if get_settings().database_enabled:
            from gaode.db.repository import get_messages

            return await get_messages(thread_id)
        return list(self._conversations.get(thread_id, []))

    async def add_message(self, thread_id: str, role: str, content: str) -> None:
        if get_settings().database_enabled:
            from gaode.db.repository import append_message

            await append_message(thread_id, role, content)
            return
        self._conversations.setdefault(thread_id, []).append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    async def clear_history(self, thread_id: str) -> None:
        if get_settings().database_enabled:
            from gaode.db.repository import delete_conversation

            await delete_conversation(thread_id)
        self._conversations.pop(thread_id, None)
        saver = await get_memory_saver()
        delete_thread = getattr(saver, "adelete_thread", None)
        if delete_thread is not None:
            await delete_thread(thread_id)


_memory_manager = MemoryManager()
_memory_saver = MemorySaver()
_postgres_pool = None
_postgres_saver = None


def get_memory_manager() -> MemoryManager:
    return _memory_manager


async def init_memory() -> None:
    """Initialize the configured LangGraph checkpoint backend."""
    global _postgres_pool, _postgres_saver
    settings = get_settings()
    if settings.checkpoint_backend != "postgres" or _postgres_saver is not None:
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    _postgres_pool = AsyncConnectionPool(
        conninfo=settings.checkpoint_database_url,
        min_size=1,
        max_size=settings.database_pool_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await _postgres_pool.open()
    _postgres_saver = AsyncPostgresSaver(_postgres_pool)
    await _postgres_saver.setup()
    logger.info("LangGraph PostgreSQL checkpointer is ready")


async def get_memory_saver():
    if get_settings().checkpoint_backend == "postgres":
        await init_memory()
        return _postgres_saver
    return _memory_saver


async def close_memory() -> None:
    global _postgres_pool, _postgres_saver
    if _postgres_pool is not None:
        await _postgres_pool.close()
    _postgres_pool = None
    _postgres_saver = None
