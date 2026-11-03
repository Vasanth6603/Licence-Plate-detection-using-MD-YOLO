"""Event processor for ALPR events."""

from datetime import datetime
from uuid import uuid4

from loguru import logger


class EventProcessor:
    """Process and deduplicate ALPR events."""

    def __init__(self, redis_client=None):
        self.redis = redis_client

    async def process_event(self, event: dict):
        """Process single ALPR event with deduplication."""
        try:
            plate = event.get("plate_number", "").upper()
            camera_id = event.get("camera_id")
            key = f"alpr:dedup:{camera_id}:{plate}"
            if self.redis:
                exists = await self.redis.get(key)
                if exists:
                    return None
                await self.redis.setex(key, 10, "1")
            event["event_id"] = event.get("event_id") or str(uuid4())
            event["processed_at"] = datetime.utcnow().isoformat()
            logger.info("Processed event: {} at {}", plate, camera_id)
            return event
        except Exception as exc:
            logger.error("Event processing error: {}", exc)
            return None
