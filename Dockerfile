FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mevin.py dashboard.html ./
RUN mkdir -p snapshots

EXPOSE 5555

ENV MEVIN_TOKEN=""
CMD ["python", "mevin.py"]
