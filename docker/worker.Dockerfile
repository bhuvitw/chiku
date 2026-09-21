# The Celery worker, and the only image that carries torch.
#
# Kept separate from backend.Dockerfile on purpose (System Design §19): the API
# serves requests and enqueues jobs, the worker runs the model. Merging them
# would put ~900 MB of CPU torch wheels into the image that handles HTTP.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY backend/requirements.txt ./backend/
COPY ml/requirements.txt ./ml/

# CPU wheels from PyTorch's own index, installed first so pip cannot resolve
# the multi-gigabyte CUDA build for a container that has no GPU.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r backend/requirements.txt -r ml/requirements.txt

COPY pyproject.toml alembic.ini ./
COPY backend/ ./backend/
COPY ml/ ./ml/

CMD ["celery", "-A", "backend.workers.celery_app", "worker", "--loglevel=info"]
