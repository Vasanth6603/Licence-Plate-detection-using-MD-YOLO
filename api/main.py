"""FastAPI application and dependency injection."""

from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from starlette.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, ModelInfoResponse
from core.config import settings
from core.detector import Detector, DetectorError

# ---------------------------------------------------------------------------
# Singleton detector instance
# ---------------------------------------------------------------------------


@lru_cache
def get_detector() -> Detector:
    """Return a cached, lazily-initialised detector.

    If weights are missing the detector is returned un-loaded so the service
    can still start and report a degraded health status.
    """
    detector = Detector()
    try:
        detector.load()
    except DetectorError:
        pass
    return detector


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the API key when one is configured."""
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key or ""


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="License Plate Detector API",
        description="Production-grade YOLOv9 license plate detection service",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        detector = get_detector()
        return HealthResponse(
            status="ok" if detector.is_loaded() else "degraded",
            version=settings.app_version,
            model_loaded=detector.is_loaded(),
            model_weights=str(settings.model_weights),
            device=settings.device,
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
    def model_info() -> ModelInfoResponse:
        detector = get_detector()
        return ModelInfoResponse(
            name="yolov9-e",
            version=settings.app_version,
            classes=list(detector._names.values()) if detector.is_loaded() else [],
            img_size=detector.img_size,
            conf_thres=detector.conf_thres,
            iou_thres=detector.iou_thres,
        )

    @app.post(
        "/predict",
        response_model=None,
        tags=["inference"],
        summary="Detect license plates in an uploaded image",
    )
    async def predict(
        file: UploadFile = File(...),
        detector: Detector = Depends(get_detector),
        _api_key: str = Security(verify_api_key),
    ):
        from api.routes.predict import handle_predict

        return await handle_predict(file, detector)

    @app.post(
        "/predict/batch",
        response_model=None,
        tags=["inference"],
        summary="Detect license plates in multiple images",
    )
    async def predict_batch(
        files: list[UploadFile] = File(...),
        detector: Detector = Depends(get_detector),
        _api_key: str = Security(verify_api_key),
    ):
        from api.routes.predict import handle_predict_batch

        return await handle_predict_batch(files, detector)

    return app


app = create_app()
