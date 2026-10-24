"""Health endpoint handler."""

from api.schemas import HealthResponse
from core.config import settings
from core.detector import get_detector


def handle_health() -> HealthResponse:
    detector = get_detector()
    return HealthResponse(
        status="ok" if detector.is_loaded() else "degraded",
        version=settings.app_version,
        model_loaded=detector.is_loaded(),
        model_weights=str(settings.model_weights),
        device=settings.device,
    )
