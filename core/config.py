"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised, typed configuration. All values overridable via env."""

    # Application
    app_name: str = Field("license-plate-detector", description="Service name")
    app_version: str = Field("1.0.0", description="Service version")
    api_host: str = Field("0.0.0.0", description="Bind host")
    api_port: int = Field(8000, description="Bind port")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")

    # Model
    model_weights: Path = Field(
        Path("models/license_plate_detector.pt"), description="Path to model weights"
    )
    model_cfg: Path = Field(
        Path("yolov9/models/detect/yolov9-e.yaml"), description="Model architecture YAML"
    )
    data_yaml: Path = Field(Path("dataset/data.yaml"), description="Dataset configuration YAML")
    img_size: int = Field(640, description="Inference image size")
    conf_thres: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_thres: float = Field(0.45, ge=0.0, le=1.0, description="NMS IoU threshold")
    max_det: int = Field(1000, ge=1, description="Max detections per image")
    device: str = Field("cpu", description="Torch device (cpu, cuda, 0, etc.)")
    half: bool = Field(False, description="Use FP16 half-precision")

    # Security
    api_key: str | None = Field(None, description="Required API key for inference")
    rate_limit_requests: int = Field(60, description="Requests per window")
    rate_limit_window: int = Field(60, description="Rate window in seconds")
    max_upload_mb: int = Field(10, description="Max upload size in MB")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("model_weights", "model_cfg", "data_yaml", mode="before")
    @classmethod
    def _resolve_paths(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


settings = get_settings()
