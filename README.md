# License Plate Detector

FastAPI service for license plate detection with optional EasyOCR recognition. The API can start without model files; in that case `/health` reports `degraded` and inference requests fail until the YOLOv9 source and model weights are available.

## Requirements

- Python 3.10 or newer
- The YOLOv9 source tree at `./yolov9` (the `models.common.DetectMultiBackend` module)
- Model weights at `./models/license_plate_detector.pt`

The YOLOv9 source and model weights are external assets and are not included in this repository. Dataset and training assets are also not required to serve inference.

## Run locally

```bash
python -m venv .venv
# Activate the virtual environment for your shell, then:
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn api.app:app --reload
```

Set configuration through environment variables or a local `.env` file. Start from `.env.example`. `API_KEY` is optional; when set, it is required on `/predict` and `/predict/batch`. `MAX_UPLOAD_MB` controls the per-image upload limit.

## Run with Docker Compose

Place the YOLOv9 source in `./yolov9` and weights in `./models/license_plate_detector.pt`, then run:

```bash
docker compose up --build
```

Compose mounts the external model assets at their expected paths. Copy `.env.example` to `.env` before starting Compose. The service is CPU-compatible by default. Configure `DEVICE` for an available accelerator only when the container runtime and installed PyTorch build support it.

## API

- `GET /health` — service and model readiness
- `GET /model/info` — model metadata
- `POST /predict` — single image inference (`multipart/form-data`, field `file`)
- `POST /predict/batch` — multiple images (`multipart/form-data`, repeated field `files`)
- `GET /docs` — interactive API documentation

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```
