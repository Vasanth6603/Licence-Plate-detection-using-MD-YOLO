# License Plate Detector - YOLOv9

Production-grade YOLOv9 license plate detection service with FastAPI.

## Structure

```
api/            FastAPI inference service (routes, schemas, deps)
core/           Config, detector wrapper, pre/post-processing
training/       Training CLI (extracted from notebook)
models/         Model weights (Git LFS / DVC - not committed raw)
data/           Raw and processed data (DVC managed)
tests/          Unit and integration tests
scripts/        Export, download, utility scripts
notebooks/      Exploration only (not production code)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your paths and API key

# 3. Download model weights
# Place license_plate_detector.pt in models/

# 4. Run the API
uvicorn api.main:app --reload

# 5. Or with Docker
docker compose up --build
```

## API

- `GET /health` - health check
- `GET /model/info` - model metadata
- `POST /predict` - single image inference (multipart file)
- `POST /predict/batch` - batch inference
- `GET /docs` - Swagger UI

## Authentication

Set `API_KEY` in `.env` and send `X-API-Key: <key>` header on inference endpoints.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

## Linting / Type Checking

```bash
ruff check .
ruff format --check .
mypy .
black --check .
isort --check .
```
