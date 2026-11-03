"""Stream manager for multiple RTSP cameras."""

import asyncio

from loguru import logger

from core.detector import Detector
from services.event_processor import EventProcessor
from services.frame_processor import FrameProcessor
from streaming.rtsp_reader import RTSPStreamReader


class StreamManager:
    """Manage multiple concurrent RTSP streams."""

    def __init__(self, detector: Detector):
        self.detector = detector
        self.streams: dict[str, RTSPStreamReader] = {}
        self.processors: dict[str, FrameProcessor] = {}
        self.event_processor = EventProcessor()
        self.running = False

    def add_stream(self, camera_id: str, rtsp_url: str):
        self.streams[camera_id] = RTSPStreamReader(rtsp_url)
        self.processors[camera_id] = FrameProcessor(self.detector, camera_id)
        logger.info("Added stream {}: {}", camera_id, rtsp_url)

    async def start_stream(self, camera_id: str):
        reader = self.streams[camera_id]
        processor = self.processors[camera_id]
        for frame in reader.read_frames(frame_skip=2):
            events = processor.process_frame(frame)
            for event in events:
                await self.event_processor.process_event(event)

    async def start_all(self):
        self.running = True
        tasks = [self.start_stream(cid) for cid in self.streams]
        await asyncio.gather(*tasks)

    def stop(self):
        self.running = False
        for reader in self.streams.values():
            reader.release()
