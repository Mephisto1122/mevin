FROM python:3.12-slim

# OpenCV runtime libs (libgl1-mesa-glx no longer exists on current Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-onvif.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-onvif.txt

COPY mevin.py mevin_worker.py mevin_core.py dashboard.html ./
COPY tools ./tools

# Database + snapshots live in one mountable folder
ENV MEVIN_DB=/app/data/monitor.db \
    MEVIN_SNAPSHOTS=/app/data/snapshots \
    MEVIN_TOKEN=""
RUN mkdir -p /app/data/snapshots
VOLUME ["/app/data"]

EXPOSE 5555
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:5555/healthz',timeout=4)" || exit 1

CMD ["python", "mevin.py"]
