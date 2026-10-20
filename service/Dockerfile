# Modified by Card Brave (2026-10): new file. Licensed AGPL-3.0-or-later.
# Build from the repository root:  docker build -f service/Dockerfile -t collectorvision-service .
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COLLECTORVISION_CACHE=/opt/collectorvision-cache \
    CV_CACHE_DIR=/opt/collectorvision-cache \
    HF_HOME=/opt/collectorvision-cache/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY collector_vision ./collector_vision
COPY service ./service
RUN pip install ".[hf]" onnxruntime ./service

# Bake models and catalog snapshots into the image so the container starts
# offline and every replica serves the same catalog version.
ARG CV_CATALOGS=mtg,pokemon,riftbound
# Retried because the catalog feed download occasionally times out.
RUN for attempt in 1 2 3; do \
      CV_CATALOGS=${CV_CATALOGS} CV_ALLOW_UNAUTHENTICATED=1 python -c \
        "from collectorvision_service import Pipeline, Settings; Pipeline(Settings.from_env())" \
      && exit 0; echo "catalog download attempt $attempt failed"; sleep 10; \
    done; exit 1

RUN useradd --system --uid 10001 vision && chown -R vision /opt/collectorvision-cache
USER vision
ENV CV_CATALOGS=${CV_CATALOGS} \
    CV_OFFLINE=1 \
    CV_PROVIDER=cpu \
    CV_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"
CMD ["python", "-m", "collectorvision_service"]
