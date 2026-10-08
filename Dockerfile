# Serving image for the cancellation scoring API.
# Build: docker build -t hotel-cancellation-api .
# Run:   docker run -p 8000:8000 hotel-cancellation-api  ->  http://localhost:8000/docs
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOTEL_ARTIFACT_DIR=/app/artifacts

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY artifacts ./artifacts

RUN useradd --create-home --uid 1000 scorer
USER scorer

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "hotel_cancellation.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
