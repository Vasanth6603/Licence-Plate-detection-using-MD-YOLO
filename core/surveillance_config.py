"""Configuration for video surveillance system."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SurveillanceSettings(BaseSettings):
    """Extended settings for real-time ALPR surveillance system."""

    # Application
    app_name: str = Field("vehicle-surveillance-alpr", description="Service name")
    app_version: str = Field("2.0.0", description="Service version")
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
    img_size: int = Field(640, description="Inference image size")
    conf_thres: float = Field(0.25, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou_thres: float = Field(0.45, ge=0.0, le=1.0, description="NMS IoU threshold")
    max_det: int = Field(1000, ge=1, description="Max detections per image")
    device: str = Field("cpu", description="Torch device")
    half: bool = Field(False, description="Use FP16")

    # Video Streaming
    frame_skip: int = Field(2, description="Process every Nth frame for efficiency")
    fps_limit: int = Field(12, description="Target FPS per stream")
    max_concurrent_streams: int = Field(64, description="Max concurrent streams")
    reconnect_delay: int = Field(5, description="Reconnect delay in seconds")
    max_reconnect_attempts: int = Field(10, description="Max reconnect attempts")

    # Tracking
    enable_tracking: bool = Field(True, description="Enable ByteTrack")
    track_buffer: int = Field(30, description="Frames to keep lost tracks")
    track_threshold: float = Field(0.25, description="Track confidence threshold")

    # OCR
    ocr_languages: list[str] = Field(default=["en"], description="OCR languages")
    ocr_gpu: bool = Field(False, description="Use GPU for OCR")
    multi_orientation: bool = Field(True, description="Enable multi-directional OCR")
    min_plate_conf: float = Field(0.3, description="Minimum OCR confidence")

    # Database
    database_url: str = Field(
        "postgresql+asyncpg://user:pass@localhost:5432/alpr_db",
        description="PostgreSQL connection URL",
    )
    timescaledb_enabled: bool = Field(True, description="Use TimescaleDB")
    db_pool_size: int = Field(20, description="Connection pool size")
    db_max_overflow: int = Field(40, description="Max overflow connections")

    # Redis
    redis_url: str = Field("redis://localhost:6379/0", description="Redis URL")
    redis_ttl: int = Field(3600, description="Default TTL in seconds")

    # Object Storage (Video Clips)
    s3_endpoint: str = Field("http://localhost:9000", description="S3/MinIO endpoint")
    s3_access_key: str = Field("minioadmin", description="S3 access key")
    s3_secret_key: str = Field("minioadmin", description="S3 secret key")
    s3_bucket_clips: str = Field("alpr-clips", description="Bucket for video clips")
    s3_use_ssl: bool = Field(False, description="Use SSL for S3")
    clip_retention_days: int = Field(2, description="Delete clips after N days")

    # Message Queue (Kafka/NATS)
    kafka_bootstrap_servers: str = Field("localhost:9092", description="Kafka brokers")
    kafka_topic_events: str = Field("alpr-events", description="Events topic")
    kafka_topic_clips: str = Field("alpr-clips", description="Clips topic")
    use_kafka: bool = Field(True, description="Use Kafka for events")

    # Security
    api_key: str | None = Field(None, description="API key")
    jwt_secret: str = Field("change-me-in-prod", description="JWT secret")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(60, description="JWT expiry")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed origins",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_surveillance_settings() -> SurveillanceSettings:
    return SurveillanceSettings()


surv_settings = get_surveillance_settings()
