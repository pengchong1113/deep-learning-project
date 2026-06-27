FROM python:3.11-slim

# mediapipe's native Tasks library dlopen's GLES/EGL even for CPU-only
# inference; opencv-python-headless still needs libglib. None of these
# ship in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libgles2 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -c "from scripts.extract_keypoints import ensure_model; ensure_model()"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
