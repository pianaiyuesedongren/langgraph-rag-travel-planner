from __future__ import annotations

import hashlib
import os
import re

from langchain_core.embeddings import Embeddings

from gaode.config import Settings, get_settings

for key, value in {
    "GRPC_VERBOSITY": "ERROR",
    "GRPC_CPP_MIN_LOG_LEVEL": "2",
    "GLOG_minloglevel": "2",
}.items():
    os.environ.setdefault(key, value)


class ConfigurationError(RuntimeError):
    """Raised when a runtime dependency or credential is missing."""


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings for offline or bootstrap use."""

    dimension = 1024

    @staticmethod
    def _tokens(text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
        if not cleaned:
            return []

        tokens: list[str] = []
        for chunk in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", cleaned):
            if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
                chars = list(chunk)
                tokens.extend(chars)
                tokens.extend(a + b for a, b in zip(chars, chars[1:], strict=False))
            else:
                tokens.append(chunk)

        return tokens or [cleaned]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokens(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big", signed=False) % self.dimension
            weight = 1.0 + (digest[4] / 255.0)
            vector[index] += weight

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ResilientEmbeddings(Embeddings):
    """Use DashScope when available, otherwise fall back to local hashes."""

    def __init__(self, primary: Embeddings | None, fallback: Embeddings):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = primary is None

    def _call(self, method: str, *args):
        if self._use_fallback or self.primary is None:
            return getattr(self.fallback, method)(*args)
        try:
            return getattr(self.primary, method)(*args)
        except Exception:
            self._use_fallback = True
            return getattr(self.fallback, method)(*args)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call("embed_documents", texts)

    def embed_query(self, text: str) -> list[float]:
        return self._call("embed_query", text)


def get_chat_model(settings: Settings | None = None):
    settings = settings or get_settings()
    api_key = settings.llm_api_key
    if not api_key:
        raise ConfigurationError(
            "Missing LLM API key. Set LLM_API_KEY, XIAOMI_API_KEY, or TOKENPLAN_API_KEY in .env."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ConfigurationError("Missing langchain-openai. Install requirements first.") from exc

    try:
        max_tokens = max(256, int(os.environ.get("LLM_MAX_TOKENS", "500")))
    except ValueError:
        max_tokens = 500

    chat_kwargs = {
        "api_key": api_key,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_retries": 1,
        "max_tokens": max_tokens,
        "streaming": True,
    }
    timeout_raw = os.environ.get("LLM_TIMEOUT_SECONDS")
    if timeout_raw:
        try:
            chat_kwargs["timeout"] = float(timeout_raw)
        except ValueError:
            pass
    if settings.llm_provider == "dashscope":
        # Qwen3 thinking mode can spend minutes on a long itinerary prompt;
        # the workflow already has dedicated planning/tool stages, so use a
        # bounded final answer generation here.
        chat_kwargs["extra_body"] = {"enable_thinking": False}

    return ChatOpenAI(
        **chat_kwargs,
    )


class DashScopeTextEmbeddings(Embeddings):
    """DashScope Text Embedding API"""

    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self.api_key = api_key

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        texts = [str(text).strip() for text in texts if str(text).strip()]
        if not texts:
            return []

        import dashscope

        os.environ["DASHSCOPE_API_KEY"] = self.api_key or ""

        embeddings: list[list[float]] = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                # This DashScope SDK version expects a plain list[str]. A
                # list of {"text": ...} objects causes the input.contents
                # error seen with the old OpenAI-compatible integration.
                input=batch,
                dimension=1024,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope embedding failed: status_code={resp.status_code}, "
                    f"code={resp.code}, message={resp.message}"
                )

            output = resp.output if isinstance(resp.output, dict) else {}
            items = output.get("embeddings", [])
            batch_vectors = [item.get("embedding") for item in items if isinstance(item, dict)]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"DashScope returned {len(batch_vectors)} embeddings for {len(batch)} texts"
                )
            embeddings.extend(batch_vectors)

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("Missing DashScope API key for embedding.")
        return self._call_api(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._call_api([text])
        if not vectors:
            raise ValueError("Cannot embed an empty query")
        return vectors[0]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    api_key = settings.dashscope_api_key or settings.llm_api_key
    fallback = LocalHashEmbeddings()
    use_remote = os.environ.get("USE_REMOTE_EMBEDDINGS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not use_remote or not api_key:
        return fallback

    return DashScopeTextEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
    )
