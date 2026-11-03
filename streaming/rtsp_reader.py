"""RTSP stream reader with reconnection logic."""

import time
from collections.abc import Generator

import cv2
from loguru import logger


class RTSPStreamReader:
    """Read frames from RTSP stream with auto-reconnect."""

    def __init__(self, rtsp_url: str, reconnect_delay: int = 5, max_attempts: int = 10):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_attempts = max_attempts
        self.cap = None
        self.attempts = 0

    def connect(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                logger.error("Failed to open RTSP stream: {}", self.rtsp_url)
                return False
            logger.info("Connected to RTSP stream: {}", self.rtsp_url)
            self.attempts = 0
            return True
        except Exception as exc:
            logger.error("RTSP connect error: {}", exc)
            return False

    def read_frames(self, frame_skip: int = 2) -> Generator:
        """Yield frames with frame skipping."""
        frame_count = 0
        while self.attempts < self.max_attempts:
            if self.cap is None or not self.cap.isOpened():
                if not self.connect():
                    self.attempts += 1
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame, reconnecting...")
                self.cap.release()
                self.cap = None
                self.attempts += 1
                time.sleep(self.reconnect_delay)
                continue

            frame_count += 1
            if frame_skip > 0 and frame_count % frame_skip != 0:
                continue

            yield frame

        logger.error("Max reconnect attempts reached for: {}", self.rtsp_url)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
