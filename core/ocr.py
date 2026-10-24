"""License Plate Recognition (OCR) module using EasyOCR."""

import re

import cv2
import numpy as np
from loguru import logger

from core.plate_rotation import get_rotated_variants, score_ocr_text


class PlateOCR:
    """OCR engine for extracting license plate numbers."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize EasyOCR reader (lazy loading)."""
        if self._initialized:
            return
        try:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            self._initialized = True
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.warning("EasyOCR not installed. OCR will be disabled.")
            self._initialized = False
        except Exception as exc:
            logger.error("Failed to initialize EasyOCR: {}", exc)
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized and self._reader is not None

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Preprocess cropped plate image for better OCR accuracy."""
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        # Apply bilateral filter to reduce noise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Upscale for better OCR
        height, width = thresh.shape[:2]
        if height < 50:
            scale = 2
            thresh = cv2.resize(
                thresh, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC
            )

        return thresh

    def clean_plate_text(self, text: str) -> str:
        """Clean and normalize OCR text."""
        # Remove special characters and spaces, keep alphanumeric
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned

    def read_plate(self, plate_img: np.ndarray) -> tuple[str, float]:
        """Extract license plate number from cropped plate image.

        Returns (plate_number, confidence).
        """
        if not self.is_available():
            return "", 0.0

        try:
            # Preprocess
            processed = self.preprocess_plate(plate_img)

            # Run OCR
            results = self._reader.readtext(processed)

            if not results:
                # Try without preprocessing
                results = self._reader.readtext(plate_img)

            if not results:
                return "", 0.0

            # Combine all detections (plates can have multiple text regions)
            best_text = ""
            best_conf = 0.0

            for _bbox, text, conf in results:
                cleaned = self.clean_plate_text(text)
                if cleaned and len(cleaned) >= 3:  # Minimum plate length
                    if conf > best_conf:
                        best_conf = conf
                        best_text = cleaned

            if not best_text and results:
                # Take highest confidence even if short
                (_bbox, text, conf) = max(results, key=lambda x: x[2])
                best_text = self.clean_plate_text(text)
                best_conf = conf

            return best_text, round(float(best_conf), 4)
        except Exception as exc:
            logger.error("OCR error: {}", exc)
            return "", 0.0

    def read_plate_best(self, plate_img: np.ndarray) -> tuple[str, float]:
        """Try multiple orientations and pick best OCR result for tilted plates."""
        if not self.is_available():
            return "", 0.0
        try:
            variants = get_rotated_variants(plate_img)
            best_text = ""
            best_conf = 0.0
            best_score = -1.0
            for variant, _ang in variants:
                try:
                    processed = self.preprocess_plate(variant)
                    results = self._reader.readtext(processed)
                    if not results:
                        results = self._reader.readtext(variant)
                    if not results:
                        continue
                    for _bbox, text, conf in results:
                        cleaned = self.clean_plate_text(text)
                        if not cleaned:
                            continue
                        c = float(conf)
                        s = score_ocr_text(cleaned, c)
                        if s > best_score:
                            best_score = s
                            best_text = cleaned
                            best_conf = c
                except Exception:
                    continue
            if best_text:
                return best_text, round(float(best_conf), 4)
            # Fallback to single pass
            return self.read_plate(plate_img)
        except Exception as exc:
            logger.error("Multi-directional OCR error: {}", exc)
            return self.read_plate(plate_img)


# Global OCR instance
ocr_engine = PlateOCR(gpu=False)
ocr_engine.initialize()
