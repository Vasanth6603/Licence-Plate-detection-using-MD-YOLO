"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, ConfigDict, Field


class DetectionOut(BaseModel):
    """A single detected license plate."""

    model_config = ConfigDict(from_attributes=True)

    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class id")
    class_name: str = Field(..., description="Class name")


class PredictResponse(BaseModel):
    """Response for a single-image prediction."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    detections: list[DetectionOut] = Field(default_factory=list)
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    inference_ms: float = Field(..., description="Inference time in milliseconds")
    num_detections: int = Field(..., description="Number of detections")


class PredictBatchResponse(BaseModel):
    """Response for batch prediction."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    results: list[PredictResponse] = Field(default_factory=list)
    total_detections: int = Field(0, description="Total detections across all images")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field("ok", description="Service status")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_weights: str = Field(..., description="Path to model weights")
    device: str = Field(..., description="Torch device in use")


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = False
    error: str = Field(..., description="Error message")
    code: str = Field("error", description="Error code")


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    classes: list[str] = Field(..., description="Class names")
    img_size: int = Field(..., description="Inference image size")
    conf_thres: float = Field(..., description="Confidence threshold")
    iou_thres: float = Field(..., description="NMS IoU threshold")
