FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY backend/requirements.txt backend/requirements-dev.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements-dev.txt

COPY pyproject.toml alembic.ini ./
COPY backend/ ./backend/
# `backend.services.uploads` imports `ml.data.deid` for the de-identification
# pass, so the API needs the ml package even though it never loads a model.
# Only the PIL/numpy-level modules are reachable from here — no torch.
COPY ml/ ./ml/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
