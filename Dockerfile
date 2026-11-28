# Image for the dental clinic assistant API.
# Build and start with Docker Compose: docker compose up --build

FROM python:3.14-slim

# PYTHONUNBUFFERED: print logs right away, so `docker logs` shows them.
# PYTHONDONTWRITEBYTECODE: no .pyc files inside the image.
# PIP_NO_CACHE_DIR: do not keep pip's download cache, the image stays smaller.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first. Docker caches each step, so this slow step is repeated
# only when requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then the code and the clinic data the index is built from.
COPY app/ app/
COPY data/ data/

# Run as a normal user, not root, so a bug in the app cannot change the system.
# /data is where the vector index lives (a volume, see compose.yaml).
RUN useradd --create-home appuser \
    && mkdir /data \
    && chown appuser /data
USER appuser
ENV CHROMA_DIR=/data/chroma

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
