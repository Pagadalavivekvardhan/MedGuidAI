"""Unit tests for backend.security module.

Tests cover:
- API key validation (get_api_key dependency)
- Rate limit key generation
- Rate limit exceeded handler
- setup_security middleware application
- API key configuration from environment
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── API Key Validation Tests ───────────────────────────────────────────────


class TestGetApiKey:
    """Tests for the get_api_key dependency function."""

    def test_valid_api_key_accepted(self):
        """A valid API key in the header should be accepted."""
        from backend.security import get_api_key

        # get_api_key is async, need to run it properly
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            get_api_key(api_key="medguid-dev-key-2024")
        )
        assert result == "medguid-dev-key-2024"

    def test_missing_api_key_raises_403(self):
        """Missing API key (None) should raise HTTPException 403."""
        from backend.security import get_api_key
        from fastapi import HTTPException

        import asyncio

        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                get_api_key(api_key=None)
            )
        assert exc_info.value.status_code == 403
        assert "Missing API key" in exc_info.value.detail

    def test_invalid_api_key_raises_403(self):
        """An invalid API key should raise HTTPException 403."""
        from backend.security import get_api_key
        from fastapi import HTTPException

        import asyncio

        with pytest.raises(HTTPException) as exc_info:
            asyncio.get_event_loop().run_until_complete(
                get_api_key(api_key="invalid-key-12345")
            )
        assert exc_info.value.status_code == 403
        assert "Invalid API key" in exc_info.value.detail


# ─── API Key Configuration Tests ────────────────────────────────────────────


class TestApiKeyConfiguration:
    """Tests for API key loading from environment variables."""

    def test_default_dev_key_used_when_no_env(self):
        """When no MEDGUID_API_KEYS is set, the default dev key should be available."""
        from backend import security

        assert "medguid-dev-key-2024" in security.API_KEYS

    def test_custom_api_keys_from_env(self):
        """Custom API keys from MEDGUID_API_KEYS env var should be loaded."""
        with patch.dict(os.environ, {"MEDGUID_API_KEYS": "key1,key2,key3"}):
            import importlib
            import backend.security
            importlib.reload(backend.security)
            assert "key1" in backend.security.API_KEYS
            assert "key2" in backend.security.API_KEYS
            assert "key3" in backend.security.API_KEYS
            # Reload to restore defaults
            importlib.reload(backend.security)

    def test_dev_key_disabled_via_env(self):
        """When MEDGUID_DISABLE_DEV_KEY=true, dev key should be excluded."""
        with patch.dict(os.environ, {"MEDGUID_DISABLE_DEV_KEY": "true"}):
            import importlib
            import backend.security
            importlib.reload(backend.security)
            assert "medguid-dev-key-2024" not in backend.security.API_KEYS
            # Reload to restore defaults
            del os.environ["MEDGUID_DISABLE_DEV_KEY"]
            importlib.reload(backend.security)


# ─── Rate Limit Key Tests ───────────────────────────────────────────────────


class TestRateLimitKey:
    """Tests for the get_rate_limit_key function."""

    def test_rate_limit_key_with_api_key(self):
        """When API key is present, rate limit key should use apikey: prefix."""
        from backend.security import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {"X-API-KEY": "test-key-123"}
        key = get_rate_limit_key(mock_request)
        assert key == "apikey:test-key-123"

    def test_rate_limit_key_without_api_key(self):
        """When no API key, rate limit key should fall back to IP address."""
        from backend.security import get_rate_limit_key, API_KEY_NAME
        from slowapi.util import get_remote_address

        mock_request = MagicMock()
        mock_request.headers = {}
        key = get_rate_limit_key(mock_request)
        # Should be the remote address (from slowapi)
        expected = get_remote_address(mock_request)
        assert key == expected


# ─── Rate Limit Handler Tests ───────────────────────────────────────────────


class TestRateLimitHandler:
    """Tests for the rate_limit_exceeded_handler function."""

    def test_rate_limit_handler_returns_429(self):
        """Rate limit exceeded should return 429 status with proper JSON."""
        from backend.security import rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded

        mock_request = MagicMock()
        # RateLimitExceeded wraps a detail string
        exc = MagicMock(spec=RateLimitExceeded)
        exc.detail = "10/minute"
        response = rate_limit_exceeded_handler(mock_request, exc)

        assert response.status_code == 429
        import json
        data = json.loads(response.body)
        assert data["error"] == "Rate limit exceeded"
        assert "retry_after" in data


# ─── Setup Security Tests ───────────────────────────────────────────────────


class TestSetupSecurity:
    """Tests for the setup_security function."""

    def test_setup_security_attaches_limiter(self):
        """setup_security should attach the limiter to app.state."""
        from backend.security import setup_security, limiter

        app = FastAPI()
        result = setup_security(app)

        assert result is app
        assert app.state.limiter is limiter

    def test_setup_security_adds_exception_handler(self):
        """setup_security should register the RateLimitExceeded handler."""
        from backend.security import setup_security

        app = FastAPI()
        setup_security(app)

        # Check that the exception handler is registered
        from slowapi.errors import RateLimitExceeded
        handlers = app.exception_handlers
        assert RateLimitExceeded in handlers


# ─── Integration Tests with TestClient ──────────────────────────────────────


class TestSecurityIntegration:
    """Integration tests using TestClient to verify security works end-to-end."""

    def _make_test_app(self):
        """Create a minimal FastAPI app with security for testing."""
        from backend.security import setup_security, get_api_key, limiter
        from fastapi import Depends

        app = FastAPI()
        setup_security(app)

        @app.get("/test-endpoint")
        async def test_endpoint(api_key: str = Depends(get_api_key)):
            return {"status": "ok", "key": api_key}

        @app.get("/unprotected")
        async def unprotected():
            return {"status": "ok"}

        return app

    def test_request_with_valid_key(self):
        """Request with valid API key should succeed."""
        app = self._make_test_app()
        client = TestClient(app)
        response = client.get(
            "/test-endpoint",
            headers={"X-API-KEY": "medguid-dev-key-2024"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_request_without_key(self):
        """Request without API key should return 403."""
        app = self._make_test_app()
        client = TestClient(app)
        response = client.get("/test-endpoint")
        assert response.status_code == 403
        assert "Missing API key" in response.json()["detail"]

    def test_request_with_invalid_key(self):
        """Request with invalid API key should return 403."""
        app = self._make_test_app()
        client = TestClient(app)
        response = client.get(
            "/test-endpoint",
            headers={"X-API-KEY": "wrong-key"}
        )
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_unprotected_endpoint_no_auth_needed(self):
        """Unprotected endpoints should not require API key."""
        app = self._make_test_app()
        client = TestClient(app)
        response = client.get("/unprotected")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
