"""Multi-directional license plate alignment utilities for better OCR."""

import cv2
import numpy as np


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image without cropping, keeping full content."""
    if image is None or image.size == 0:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (n_w / 2) - center[0]
    matrix[1, 2] += (n_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (n_w, n_h), borderValue=(0, 0, 0))


def deskew_by_minarearect(plate_img: np.ndarray) -> np.ndarray:
    """Attempt deskew using minAreaRect on edge-detected plate."""
    if plate_img is None or plate_img.size == 0:
        return plate_img
    try:
        gray = (
            cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        )
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) < 50:
            return plate_img
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 15:
            return plate_img
        return rotate_bound(plate_img, angle)
    except Exception:
        return plate_img


def get_rotated_variants(
    plate_img: np.ndarray, angles: list[float] | None = None
) -> list[tuple[np.ndarray, float]]:
    """Generate rotated variants of plate crop."""
    if plate_img is None or plate_img.size == 0:
        return [(plate_img, 0.0)]
    if angles is None:
        angles = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    variants = []
    # Original
    variants.append((plate_img.copy(), 0.0))
    # Deskew attempt
    try:
        deskewed = deskew_by_minarearect(plate_img)
        if deskewed is not plate_img:
            variants.append((deskewed, 0.0))
    except Exception:
        pass
    # Rotated variants
    for ang in angles:
        if ang == 0:
            continue
        try:
            rot = rotate_bound(plate_img, ang)
            variants.append((rot, ang))
        except Exception:
            continue
    # Remove duplicates by shape
    unique = []
    seen = set()
    for img, ang in variants:
        key = (img.shape[0], img.shape[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append((img, ang))
    return unique


def score_ocr_text(text: str, conf: float) -> float:
    """Score OCR result to prefer better plate reads."""
    if not text:
        return 0.0
    cleaned = "".join(c for c in text.upper() if c.isalnum())
    if not cleaned:
        return 0.0
    # Prefer longer alphanumeric strings (typical plates 6-10 chars)
    score = len(cleaned) * 10.0
    score += conf * 100.0
    # Bonus for reasonable plate length
    if 4 <= len(cleaned) <= 12:
        score += 20.0
    return score
