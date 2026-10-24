"""Shared API dependencies."""

from fastapi import Security
from fastapi.security import APIKeyHeader

from core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Validate the API key when one is configured."""
    if settings.api_key and api_key != settings.api_key:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key or ""
