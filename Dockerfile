# ---- builder -------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# PyAV needs the FFmpeg dev headers to build; the runtime stage only needs the
# shared libraries, which is why this is a two-stage build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY firemex ./firemex

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# ONNX Runtime rather than torch: 2-4x the throughput on the same hardware and it
# keeps the image roughly 2 GB smaller. Export weights with
#   yolo export model=weights/firemex.pt format=onnx imgsz=640 dynamic=True
RUN pip install --upgrade pip && pip install ".[video,onnx]"

# ---- runtime -------------------------------------------------------------
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FIREMEX_STORAGE_DIR=/data \
    FIREMEX_CONFIG_PATH=/config/config.yaml \
    FIREMEX_MODEL_PATH=/weights/firemex.onnx \
    FIREMEX_DETECTOR_BACKEND=onnx

RUN apt-get update && apt-get install -y --no-install-recommends \
        libavformat61 libavcodec61 libavdevice61 libavutil59 \
        libavfilter10 libswscale8 libswresample5 \
        curl tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Non-root: the process needs no privileges beyond outbound network and /data.
RUN useradd --create-home --uid 10001 firemex \
    && mkdir -p /data /config /weights \
    && chown -R firemex:firemex /data /config
USER firemex
WORKDIR /home/firemex

VOLUME ["/data"]
EXPOSE 8000

# Liveness only. Readiness (/api/ready) reports camera health and is deliberately
# not wired here: a single dropped camera should page someone, not restart the
# container and take every other camera down with it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["firemex", "serve"]
