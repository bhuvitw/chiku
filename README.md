# AI-Assisted Musculoskeletal Imaging Platform

Research and educational prototype. **Not a medical device, not FDA/CE cleared,
and not for clinical use.** No output from this system is a diagnosis.

- [`prd.md`](prd.md) — what it does and explicitly does not do
- [`System Design — AI-Assisted Musculoskeletal Imaging Platform.md`](System%20Design%20—%20AI-Assisted%20Musculoskeletal%20Imaging%20Platform.md) — architecture
- [`implementation-plan.md`](implementation-plan.md) — phased build order and exit criteria

## Status

**Phase 0 — scaffolding.** No ML yet: the model work starts in Phase 1, and the
architecture deliberately does not run ahead of it (System Design §31).

## Running it

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health |

The backend applies migrations on start, so the first boot creates the schema.

### Without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
# Postgres and Redis must be reachable per .env
.venv/bin/alembic upgrade head
.venv/bin/uvicorn backend.main:app --reload

cd frontend && npm install && npm run dev
```

## Tests

```bash
.venv/bin/pytest          # runs against Postgres, not SQLite — see backend/tests/conftest.py
.venv/bin/ruff check backend
cd frontend && npm run lint && npm run build
```

## Layout

```
backend/    FastAPI app, SQLAlchemy models, Celery workers, Alembic migrations
frontend/   Next.js app (App Router, Tailwind)
ml/         training/evaluation code — empty until Phase 1
data/       DVC-tracked datasets; nothing here is committed to Git
docker/     container definitions
```
# chiku
