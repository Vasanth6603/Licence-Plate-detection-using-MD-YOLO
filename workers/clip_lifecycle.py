"""Lifecycle worker to delete video clips after retention period."""

import asyncio

from loguru import logger


class ClipLifecycleWorker:
    """Automated deletion of video clips after 2 days, retain metadata."""

    def __init__(self, retention_days: int = 2):
        self.retention_days = retention_days
        self.running = False

    async def run_once(self):
        """Run one lifecycle cleanup cycle."""
        logger.info("Starting clip lifecycle cleanup")
        # In production: query DB for expired clips, delete from S3/MinIO, update DB
        # For now, log intent
        logger.info(f"Checking for clips older than {self.retention_days} days")
        return {"deleted": 0, "errors": 0}

    async def start(self, interval_minutes: int = 30):
        """Start continuous cleanup worker."""
        self.running = True
        while self.running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("Lifecycle worker error: {}", exc)
            await asyncio.sleep(interval_minutes * 60)

    def stop(self):
        self.running = False


async def main():
    worker = ClipLifecycleWorker(retention_days=2)
    await worker.run_once()


if __name__ == "__main__":
    asyncio.run(main())
