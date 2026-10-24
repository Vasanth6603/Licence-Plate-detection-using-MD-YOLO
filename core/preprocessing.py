"""Image preprocessing utilities for license plate detection."""

from pathlib import Path

import cv2
import numpy as np


class PreprocessingError(Exception):
    """Raised when an image cannot be preprocessed."""


def load_image(source: str | Path | bytes) -> np.ndarray:
    """Load an image from path, PIL-compatible bytes, or raw bytes.

    Returns a BGR numpy array (OpenCV convention).
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise PreprocessingError(f"Image file not found: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise PreprocessingError(f"Failed to decode image: {path}")
            return image

        if isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise PreprocessingError("Failed to decode image bytes")
            return image

        raise PreprocessingError(f"Unsupported image source type: {type(source)}")
    except PreprocessingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise PreprocessingError(f"Unexpected error loading image: {exc}") from exc


def validate_image(image: np.ndarray, max_dimension: int = 4096) -> None:
    """Validate image dimensions and channel count."""
    if image is None or image.size == 0:
        raise PreprocessingError("Image is empty")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise PreprocessingError("Image must be a 3-channel RGB/BGR image")
    height, width = image.shape[:2]
    if height > max_dimension or width > max_dimension:
        raise PreprocessingError(f"Image dimensions {width}x{height} exceed max {max_dimension}")


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize image to a multiple of stride while preserving aspect ratio.

    Returns the resized image, the gain (scale factor), and the padding (dw, dh).
    This mirrors YOLO's letterbox behaviour used during training.
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def preprocess(
    image: np.ndarray,
    img_size: int = 640,
    stride: int = 32,
    half: bool = False,
) -> np.ndarray:
    """Preprocess an image for YOLO inference.

    Returns a float32 (or float16) CHW tensor normalised to [0, 1], batched to NCHW.
    """
    validate_image(image)
    img, _, _ = letterbox(image, new_shape=(img_size, img_size), stride=stride)

    # BGR to RGB, HWC to CHW, to tensor
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)

    dtype = np.float16 if half else np.float32
    img = img.astype(dtype)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)  # add batch dimension
    return img
