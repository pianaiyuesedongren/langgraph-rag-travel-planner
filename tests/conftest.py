import os

os.environ.setdefault("DATABASE_ENABLED", "false")
os.environ.setdefault("CHECKPOINT_BACKEND", "memory")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("VECTOR_STORE_ENABLED", "false")
os.environ.setdefault("USE_REMOTE_EMBEDDINGS", "false")
os.environ.setdefault("USE_LLM_SUPERVISOR", "false")
os.environ.setdefault("USE_REACT_AGENTS", "false")
os.environ.setdefault("USE_LLM_ITINERARY", "false")
