"""Frame processor for real-time ALPR."""

from datetime import datetime
from uuid import uuid4

import numpy as np
from loguru import logger

from core.detector import Detector
from core.ocr import ocr_engine


class FrameProcessor:
    """Process video frames for license plate detection and recognition."""

    def __init__(self, detector: Detector, camera_id: str):
        self.detector = detector
        self.camera_id = camera_id
        self.track_history = {}

    def process_frame(self, frame: np.ndarray):
        """Process single frame and return events."""
        try:
            result = self.detector.predict(frame)
            events = []
            for det in result.detections:
                plate_number = ""
                ocr_conf = 0.0
                if ocr_engine.is_available():
                    try:
                        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                        pad = 5
                        h, w = frame.shape[:2]
                        x1_p = max(0, x1 - pad)
                        y1_p = max(0, y1 - pad)
                        x2_p = min(w, x2 + pad)
                        y2_p = min(h, y2 + pad)
                        plate_crop = frame[y1_p:y2_p, x1_p:x2_p]
                        if plate_crop.size > 0:
                            plate_number, ocr_conf = ocr_engine.read_plate_best(plate_crop)
                    except Exception as exc:
                        logger.warning("OCR failed: {}", exc)

                if plate_number and len(plate_number) >= 3:
                    event = {
                        "event_id": str(uuid4()),
                        "plate_number": plate_number,
                        "plate_normalized": plate_number.upper(),
                        "camera_id": self.camera_id,
                        "detected_at": datetime.utcnow().isoformat(),
                        "det_conf": round(float(det.confidence), 4),
                        "ocr_conf": round(float(ocr_conf), 4),
                        "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                    }
                    events.append(event)
            return events
        except Exception as exc:
            logger.error("Frame processing error: {}", exc)
            return []
