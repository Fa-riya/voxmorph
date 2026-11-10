# --- Stage 1: build the React frontend ------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python service that also serves the built frontend ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOXMORPH_STATIC_DIR=/app/static
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY --from=frontend /frontend/dist /app/static

EXPOSE 8000
CMD ["uvicorn", "voxmorph.api:app", "--host", "0.0.0.0", "--port", "8000"]
