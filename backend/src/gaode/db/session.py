from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from gaode.config import get_settings
from gaode.db.models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_session_factory() -> async_sessionmaker | None:
    global _engine, _session_factory
    settings = get_settings()
    if not settings.database_enabled:
        return None
    if _session_factory is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory


async def init_database() -> None:
    settings = get_settings()
    factory = get_session_factory()
    if factory is None or _engine is None:
        logger.info("PostgreSQL business persistence is disabled")
        return
    if settings.database_auto_create:
        async with _engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    async with _engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    logger.info("PostgreSQL business persistence is ready")


async def database_health() -> dict[str, bool]:
    settings = get_settings()
    if not settings.database_enabled:
        return {"enabled": False, "available": False}
    try:
        factory = get_session_factory()
        if factory is None:
            return {"enabled": True, "available": False}
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return {"enabled": True, "available": True}
    except Exception:
        logger.exception("PostgreSQL health check failed")
        return {"enabled": True, "available": False}


async def close_database() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
