from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from gaode.config import Settings, get_settings


class RedisManager:
    """Small Redis Streams/cache facade with graceful local fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        self._available: bool | None = None
        self._local_cache: dict[str, tuple[float, str]] = {}

    async def _get_client(self):
        if not self.settings.redis_enabled:
            return None
        if self._client is None:
            try:
                from redis.asyncio import Redis

                self._client = Redis.from_url(
                    self.settings.redis_url,
                    password=self.settings.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=5,
                )
            except Exception:
                self._available = False
                return None

        try:
            await self._client.ping()
            self._available = True
            return self._client
        except Exception:
            self._available = False
            return None

    @property
    def available(self) -> bool:
        return self._available is True

    async def health(self) -> dict[str, Any]:
        client = await self._get_client()
        if client is None:
            return {"enabled": self.settings.redis_enabled, "available": False}
        info = await client.info(section="server")
        return {
            "enabled": True,
            "available": True,
            "version": info.get("redis_version"),
        }

    @staticmethod
    def request_key(question: str, thread_id: str | None, use_memory: bool) -> str:
        raw = json.dumps(
            {"question": question.strip(), "thread_id": thread_id, "use_memory": use_memory},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get_json(self, key: str) -> Any | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            value = await client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None

    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.set(
                key,
                json.dumps(value, ensure_ascii=False, default=str),
                ex=ttl or self.settings.redis_cache_ttl,
            )
            return True
        except Exception:
            return False

    async def acquire_lock(self, key: str, token: str, ttl: int = 600) -> bool:
        client = await self._get_client()
        if client is None:
            return True
        try:
            return bool(await client.set(key, token, ex=ttl, nx=True))
        except Exception:
            return True

    async def release_lock(self, key: str, token: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        script = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
        try:
            await client.eval(script, 1, key, token)
        except Exception:
            pass

    async def publish_event(self, job_id: str, event: dict[str, Any]) -> str | None:
        client = await self._get_client()
        if client is None:
            return None
        key = f"travel:events:{job_id}"
        try:
            event_id = await client.xadd(
                key,
                {"event": json.dumps(event, ensure_ascii=False, default=str)},
                maxlen=2000,
                approximate=True,
            )
            await client.expire(key, self.settings.redis_event_ttl)
            return event_id
        except Exception:
            return None

    async def enqueue_job(self, payload: dict[str, Any]) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            await client.lpush(
                "travel:jobs",
                json.dumps(payload, ensure_ascii=False, default=str),
            )
            return True
        except Exception:
            return False

    async def dequeue_job(self, timeout: int = 5) -> dict[str, Any] | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            item = await client.brpop("travel:jobs", timeout=timeout)
            return json.loads(item[1]) if item else None
        except Exception:
            return None

    async def read_events(
        self,
        job_id: str,
        last_id: str = "0-0",
        block_ms: int = 15000,
    ) -> list[tuple[str, dict[str, Any]]]:
        client = await self._get_client()
        if client is None:
            return []
        try:
            rows = await client.xread(
                {f"travel:events:{job_id}": last_id},
                count=50,
                block=block_ms,
            )
            events: list[tuple[str, dict[str, Any]]] = []
            for _, messages in rows:
                for event_id, fields in messages:
                    events.append((event_id, json.loads(fields["event"])))
            return events
        except Exception:
            return []

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


@lru_cache(maxsize=1)
def get_redis_manager() -> RedisManager:
    return RedisManager()
