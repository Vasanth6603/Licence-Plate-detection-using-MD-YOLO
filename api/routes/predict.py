"""Single-image prediction endpoint."""

from fastapi import HTTPException, UploadFile, status
from loguru import logger

from api.schemas import PredictResponse
from core.detector import Detector
from core.postprocessing import InferenceResult
from core.preprocessing import PreprocessingError, load_image


async def handle_predict(
    file: UploadFile,
    detector: Detector,
):
    """Detect license plates in a single uploaded image."""
    # Validate content type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image",
        )

    # Read bytes with size limit
    max_bytes = 10 * 1024 * 1024  # 10 MB, overridable via settings
    try:
        contents = await file.read()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read upload: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file",
        ) from None

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {max_bytes // (1024 * 1024)} MB limit",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Load and validate image
    try:
        image = load_image(contents)
    except PreprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    # Run inference
    try:
        result: InferenceResult = detector.predict(image)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Inference error: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        ) from None

    return PredictResponse(
        detections=[d.to_dict() for d in result.detections],
        image_width=result.image_width,
        image_height=result.image_height,
        inference_ms=result.inference_ms,
        num_detections=result.num_detections,
    )


async def handle_predict_batch(files: list, detector: Detector):
    """Detect license plates in multiple uploaded images."""
    results = []
    total = 0
    for file in files:
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": "Not an image",
                }
            )
            continue

        try:
            contents = await file.read()
            image = load_image(contents)
            result: InferenceResult = detector.predict(image)
            results.append(
                {
                    "filename": file.filename,
                    "success": True,
                    "detections": [d.to_dict() for d in result.detections],
                    "image_width": result.image_width,
                    "image_height": result.image_height,
                    "inference_ms": result.inference_ms,
                    "num_detections": result.num_detections,
                }
            )
            total += result.num_detections
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": str(exc),
                }
            )

    return {
        "success": True,
        "results": results,
        "total_detections": total,
    }
