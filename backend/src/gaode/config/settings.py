from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    bailian_api_key: str | None
    dashscope_api_key: str | None
    amap_maps_api_key: str | None
    embedding_provider: str
    embedding_model: str
    vector_store_enabled: bool
    redis_url: str
    redis_password: str | None
    redis_enabled: bool
    redis_cache_ttl: int
    redis_event_ttl: int
    database_enabled: bool
    database_url: str
    checkpoint_backend: str
    checkpoint_database_url: str
    database_pool_size: int
    database_max_overflow: int
    database_auto_create: bool

    @property
    def llm_api_key(self) -> str | None:
        if self.llm_provider == "dashscope":
            return (
                self.bailian_api_key
                or self.dashscope_api_key
                or _first_env(
                    "LLM_API_KEY", "XIAOMI_API_KEY", "TOKENPLAN_API_KEY", "OPENAI_API_KEY"
                )
            )
        return _first_env("LLM_API_KEY", "XIAOMI_API_KEY", "TOKENPLAN_API_KEY", "OPENAI_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[4]
    _load_dotenv(root / ".env")

    return Settings(
        app_name=os.environ.get("APP_NAME", "Gaode Travel Planner"),
        environment=os.environ.get("APP_ENV", "local"),
        llm_provider=os.environ.get("LLM_PROVIDER", "openai_compatible").lower(),
        llm_base_url=os.environ.get(
            "LLM_BASE_URL",
            "https://token-plan-cn.xiaomimimo.com/v1",
        ),
        llm_model=os.environ.get("LLM_MODEL", "mimo-v2.5"),
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
        bailian_api_key=os.environ.get("BAILIAN_API_KEY"),
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        amap_maps_api_key=os.environ.get("AMAP_MAPS_API_KEY"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "dashscope").lower(),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v3"),
        vector_store_enabled=os.environ.get("VECTOR_STORE_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"},
        redis_url=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
        redis_password=os.environ.get("REDIS_PASSWORD") or None,
        redis_enabled=os.environ.get("REDIS_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        redis_cache_ttl=int(os.environ.get("REDIS_CACHE_TTL", "3600")),
        redis_event_ttl=int(os.environ.get("REDIS_EVENT_TTL", "86400")),
        database_enabled=os.environ.get("DATABASE_ENABLED", "false").lower()
        in {"1", "true", "yes", "on"},
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://gaode:gaode@127.0.0.1:5432/gaode",
        ),
        checkpoint_backend=os.environ.get("CHECKPOINT_BACKEND", "memory").lower(),
        checkpoint_database_url=os.environ.get(
            "CHECKPOINT_DATABASE_URL",
            "postgresql://gaode:gaode@127.0.0.1:5432/gaode",
        ),
        database_pool_size=max(1, int(os.environ.get("DATABASE_POOL_SIZE", "5"))),
        database_max_overflow=max(0, int(os.environ.get("DATABASE_MAX_OVERFLOW", "10"))),
        database_auto_create=os.environ.get("DATABASE_AUTO_CREATE", "false").lower()
        in {"1", "true", "yes", "on"},
    )
