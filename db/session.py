"""Database session management."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.surveillance_config import surv_settings

engine = create_async_engine(
    surv_settings.database_url,
    echo=surv_settings.debug,
    pool_size=surv_settings.db_pool_size,
    max_overflow=surv_settings.db_max_overflow,
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
