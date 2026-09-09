# Gaode LangGraph Travel Planner

面向真实旅行场景的多智能体规划系统。系统使用 LangGraph 编排需求解析、景点检索、餐饮推荐、路线规划和行程生成，通过高德地图 MCP 获取实时地点与路线信息，并用 RAG 知识库补充适老、亲子、门票和餐饮偏好等领域知识。

## 核心能力

- LangGraph 多 Agent 工作流，路线与餐饮节点并行执行
- DeepSeek/OpenAI-compatible LLM，可通过环境变量切换模型
- 高德地图 MCP 工具调用
- DashScope Embedding + Milvus Lite 向量检索
- PostgreSQL 持久化会话、计划、消息和 Agent 运行记录
- PostgreSQL LangGraph Checkpointer，支持多实例持久化对话状态
- Redis 缓存、异步任务队列和可重放事件流
- SSE Token 流式输出和任务进度
- RAG 来源结构化返回与前端展示
- Docker Compose 一键启动、Alembic 迁移、Pytest 和 GitHub Actions CI

## 系统架构

```text
Vue 3 / Nginx
      │ HTTP + SSE
FastAPI API ───────── PostgreSQL
      │                 ├─ conversations/messages/plans
      │                 └─ LangGraph checkpoints
      ├─────────────── Redis
      │                 ├─ cache
      │                 ├─ job queue
      │                 └─ event streams
      └── LangGraph workflow
          ├─ Supervisor
          ├─ Attraction Agent ── RAG + AMap MCP
          ├─ Route Agent ─────── AMap MCP
          ├─ Dining Agent ────── RAG + AMap MCP
          └─ Itinerary Agent ─── LLM
```

## 本地开发

项目唯一源码位于 Windows 工作区；WSL 中复用已有 Python 环境：

```bash
source /home/ricardo/Gaode_Langgraph/venv/bin/activate
cd /mnt/c/Users/Ricardo/PycharmProjects/Gaode_Langgraph
python app.py
```

前端在 Windows PowerShell 中启动：

```powershell
cd C:\Users\Ricardo\PycharmProjects\Gaode_Langgraph\frontend
npm install
npm run dev -- --host 0.0.0.0
```

访问 <http://127.0.0.1:5173>，后端健康检查为 <http://127.0.0.1:8000/api/health>。

## Docker Compose

复制配置并填写 LLM、高德和 Embedding 密钥：

```bash
cp .env.example .env
docker compose up --build
```

Compose 会启动 PostgreSQL、Redis、API、异步 Worker 和 Nginx 前端。API 启动前自动执行 Alembic 数据库迁移。

## 配置说明

| 配置 | 用途 |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | OpenAI-compatible 模型 |
| `EMBEDDING_MODEL` / `DASHSCOPE_API_KEY` | DashScope 向量模型 |
| `AMAP_MAPS_API_KEY` | 高德地图 MCP |
| `DATABASE_ENABLED` / `DATABASE_URL` | 业务数据持久化 |
| `CHECKPOINT_BACKEND` | `memory` 或 `postgres` |
| `REDIS_URL` / `REDIS_PASSWORD` | 缓存、任务和事件 |
| `USE_REACT_AGENTS` | 是否启用 ReAct + MCP Agent |
| `USE_REMOTE_EMBEDDINGS` | 是否使用远程 Embedding |

本地没有 PostgreSQL 时保留 `DATABASE_ENABLED=false` 和 `CHECKPOINT_BACKEND=memory`。Docker Compose 会覆盖为 PostgreSQL 模式。

## 数据库迁移

```bash
export MIGRATION_DATABASE_URL=postgresql+psycopg://gaode:password@127.0.0.1:5432/gaode
alembic upgrade head
```

新增模型后生成迁移：

```bash
alembic revision --autogenerate -m "describe change"
```

## 知识库导入与评测

```bash
python scripts/ingest_docs.py --dir resource/knowledge/attractions --type attraction
python scripts/ingest_docs.py --dir resource/knowledge/restaurants --type restaurant
python scripts/evaluate_retrieval.py -k 3
```

批量生成前复制 `docs/knowledge-document-template.md`，将新文档放入
`resource/knowledge/generated`，生成后先执行 `python scripts/validate_knowledge.py`。
每篇文档必须包含统一标题、城市、来源、核验日期、地址、评分、类别及领域字段。AI 生成的数据必须经过来源核验，不能把未经验证的营业时间、价格和地址作为事实入库。

## 质量检查

```bash
ruff check app.py backend/src tests scripts migrations
ruff format --check app.py backend/src tests scripts migrations
pytest --cov=gaode --cov-report=term-missing
cd frontend && npm ci && npm run build
```

## API

- `POST /api/plan`：同步返回完整计划
- `POST /api/plan/stream`：SSE 流式规划
- `POST /api/plan/jobs`：创建后台任务
- `GET /api/plan/stream/{job_id}`：重放任务事件
- `GET /api/history/{thread_id}`：读取会话历史
- `DELETE /api/history/{thread_id}`：删除会话及 Checkpoint
- `GET /api/health`：服务健康状态

## 当前边界

- Milvus Lite 仍用于本地向量检索；后续计划迁移到 PostgreSQL + pgvector，或在大规模数据下改用独立 Milvus。
- Redis 队列尚未实现 ACK、死信队列和崩溃恢复。
- 用户认证、权限隔离、OpenTelemetry 和公开云部署仍在后续阶段。
- RAG 评测集目前较小，指标只能作为回归基线，不能代表线上质量。
