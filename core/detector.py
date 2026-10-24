"""YOLOv9 model wrapper for license plate detection."""

import time
from pathlib import Path

import numpy as np
import torch

from core.config import settings
from core.postprocessing import InferenceResult, format_result
from core.preprocessing import preprocess


class DetectorError(Exception):
    """Raised when the detector fails to load or run inference."""


class Detector:
    """Production wrapper around a YOLOv9 DetectMultiBackend."""

    def __init__(
        self,
        weights: Path | None = None,
        device: str | None = None,
        half: bool | None = None,
        img_size: int | None = None,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        max_det: int | None = None,
    ):
        self.weights = weights or settings.model_weights
        self.device = device or settings.device
        self.half = half if half is not None else settings.half
        self.img_size = img_size or settings.img_size
        self.conf_thres = conf_thres if conf_thres is not None else settings.conf_thres
        self.iou_thres = iou_thres if iou_thres is not None else settings.iou_thres
        self.max_det = max_det or settings.max_det

        self._model = None
        self._stride = 32
        self._names: dict[int, str] = {}
        self._pt = False
        self._loaded = False

    def load(self) -> "Detector":
        """Load the model weights into memory. Call once at startup."""
        if self._loaded:
            return self

        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yolov9"))

            from models.common import DetectMultiBackend  # noqa: WPS433

            if not self.weights.exists():
                raise DetectorError(f"Weights file not found: {self.weights}")

            device = torch.device(self.device)
            self._model = DetectMultiBackend(
                str(self.weights),
                device=device,
                dnn=False,
                data=settings.data_yaml,
                fp16=self.half,
            )
            self._stride = int(self._model.stride)
            self._names = self._model.names
            self._pt = self._model.pt
            self._model.eval()
            self._loaded = True
        except ImportError as exc:  # pragma: no cover - environment issue
            raise DetectorError(
                "YOLOv9 package not available. Ensure the yolov9 directory is present."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise DetectorError(f"Failed to load model: {exc}") from exc
        return self

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        max_det: int | None = None,
    ) -> InferenceResult:
        """Run inference on a single BGR image (numpy array)."""
        if not self._loaded:
            self.load()

        conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        iou_thres = iou_thres if iou_thres is not None else self.iou_thres
        max_det = max_det or self.max_det

        try:
            # Preprocess
            input_tensor = preprocess(
                image,
                img_size=self.img_size,
                stride=self._stride,
                half=self.half,
            )

            # Inference
            start = time.perf_counter()
            with torch.no_grad():
                torch_input = torch.from_numpy(input_tensor).to(self._model.device)
                if self.half and torch_input.dtype != torch.float16:
                    torch_input = torch_input.half()
                pred = self._model(torch_input)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Postprocess
            result = format_result(
                raw_pred=pred.cpu().numpy(),
                orig_shape=image.shape,
                names=self._names,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                inference_ms=elapsed_ms,
            )
            return result
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise DetectorError(f"Inference failed: {exc}") from exc

    def is_loaded(self) -> bool:
        return self._loaded
