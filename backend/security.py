"""Security module for MedGuid API.

Provides API key authentication and rate limiting for all endpoints.
"""

import os
from fastapi import Security, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_403_FORBIDDEN

# ─── API Key Configuration ──────────────────────────────────────────────────

API_KEY_NAME = "X-API-KEY"

# Get API keys from environment (comma-separated for multiple keys)
API_KEYS = os.getenv("MEDGUID_API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]

# If no API keys configured, use a default development key (can be disabled in production)
if not API_KEYS:
    # Check if dev key should be disabled
    if os.getenv("MEDGUID_DISABLE_DEV_KEY", "false").lower() == "true":
        API_KEYS = []
        import logging
        logging.getLogger(__name__).warning(
            "No MEDGUID_API_KEYS set and dev key disabled. Authentication will reject all requests."
        )
    else:
        API_KEYS = ["medguid-dev-key-2024"]
        import logging
        logging.getLogger(__name__).warning(
            "No MEDGUID_API_KEYS environment variable set. Using default development key."
        )

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# ─── Authentication Dependency ───────────────────────────────────────────────

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key from request header.
    
    Args:
        api_key: The API key from the X-API-KEY header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    if api_key is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Missing API key. Include 'X-API-KEY' header."
        )
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )
    
    return api_key


# ─── Rate Limiting Configuration ─────────────────────────────────────────────

def get_rate_limit_key(request: Request) -> str:
    """Get the rate limit key from the request.
    
    Uses API key if available, otherwise falls back to IP address.
    This ensures rate limiting is per-user, not per-IP.
    """
    api_key = request.headers.get(API_KEY_NAME)
    if api_key:
        return f"apikey:{api_key}"
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=["100/minute"],  # Default: 100 requests per minute per user
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Too many requests. Limit: {exc.detail}",
            "retry_after": "60 seconds"
        }
    )


# ─── Apply to App ───────────────────────────────────────────────────────────

def setup_security(app):
    """Apply security middleware to the FastAPI app.
    
    Args:
        app: The FastAPI application instance
    """
    # Add rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    
    return app
