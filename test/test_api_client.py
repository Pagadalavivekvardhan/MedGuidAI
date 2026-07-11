"""Unit tests for frontend.api_client module.

Tests cover:
- get_backend_url, get_api_key, get_headers (session state helpers)
- check_backend_health
- upload_prescription
- analyze_lab_report
- send_chat_message
- get_diet_quick_suggestions
- get_diet_personalized_plan
- Error handling (403, 429, 500, connection errors)
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Session State Helper Tests ─────────────────────────────────────────────


class TestSessionHelpers:
    """Tests for get_backend_url, get_api_key, get_headers."""

    def test_get_backend_url_returns_default(self):
        """Should return default URL when session state has no backend_url."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {}
            from frontend.api_client import get_backend_url
            url = get_backend_url()
            assert url == "http://localhost:8000"

    def test_get_backend_url_returns_custom(self):
        """Should return custom URL from session state."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {"backend_url": "http://myserver:9000"}
            from frontend.api_client import get_backend_url
            url = get_backend_url()
            assert url == "http://myserver:9000"

    def test_get_api_key_returns_empty_when_not_set(self):
        """Should return empty string when no API key in session state."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {}
            from frontend.api_client import get_api_key
            key = get_api_key()
            assert key == ""

    def test_get_api_key_returns_none_when_key_not_present(self):
        """Session state without 'api_key' key at all should return empty string (not None)."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {"backend_url": "http://test:8000"}
            from frontend.api_client import get_api_key
            key = get_api_key()
            assert key is not None
            assert key == ""

    def test_get_api_key_returns_key(self):
        """Should return API key from session state."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {"api_key": "my-secret-key"}
            from frontend.api_client import get_api_key
            key = get_api_key()
            assert key == "my-secret-key"

    def test_get_headers_with_api_key(self):
        """Headers should include X-API-KEY when key is set."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {"api_key": "my-key"}
            from frontend.api_client import get_headers
            headers = get_headers()
            assert headers["X-API-KEY"] == "my-key"
            assert headers["Content-Type"] == "application/json"

    def test_get_headers_without_api_key(self):
        """Headers should not include X-API-KEY when no key is set."""
        with patch("frontend.api_client.st") as mock_st:
            mock_st.session_state = {}
            from frontend.api_client import get_headers
            headers = get_headers()
            assert "X-API-KEY" not in headers
            assert headers["Content-Type"] == "application/json"


# ─── Health Check Tests ─────────────────────────────────────────────────────


class TestCheckBackendHealth:
    """Tests for check_backend_health function."""

    def test_health_check_success(self):
        """Should return True when backend returns 200."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_requests.get.return_value = mock_resp
            mock_requests.exceptions = MagicMock()

            from frontend.api_client import check_backend_health
            result = check_backend_health()
            assert result is True

    def test_health_check_failure(self):
        """Should return False when backend returns non-200."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_requests.get.return_value = mock_resp
            mock_requests.exceptions = MagicMock()

            from frontend.api_client import check_backend_health
            result = check_backend_health()
            assert result is False

    def test_health_check_connection_error(self):
        """Should return False when connection fails."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_requests.get.side_effect = ConnectionError("Connection refused")
            mock_requests.exceptions = MagicMock()
            mock_requests.exceptions.RequestException = Exception

            from frontend.api_client import check_backend_health
            result = check_backend_health()
            assert result is False


# ─── Upload Prescription Tests ──────────────────────────────────────────────


class TestUploadPrescription:
    """Tests for upload_prescription function."""

    def test_upload_empty_file_bytes(self):
        """Should still attempt upload with empty file bytes (server handles validation)."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.json.return_value = {"detail": "Invalid image"}
            mock_resp.text = "error"
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import upload_prescription
            with pytest.raises(Exception, match="Invalid image"):
                upload_prescription(b"", "test.png")

            # Verify the call was made with the empty bytes
            files = mock_requests.post.call_args.kwargs["files"]
            assert files["file"][1] == b""

    def test_upload_success(self):
        """Should return medicines dict on successful upload."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"medicines": [{"name": "Test"}]}
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import upload_prescription
            result = upload_prescription(b"fake-image", "test.png")
            assert "medicines" in result
            assert len(result["medicines"]) == 1

    def test_upload_403_raises_error(self):
        """Should raise exception on 403 (invalid API key)."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import upload_prescription
            with pytest.raises(Exception, match="Invalid or missing API key"):
                upload_prescription(b"fake-image", "test.png")

    def test_upload_429_raises_error(self):
        """Should raise exception on 429 (rate limited)."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import upload_prescription
            with pytest.raises(Exception, match="Rate limit exceeded"):
                upload_prescription(b"fake-image", "test.png")

    def test_upload_500_raises_error(self):
        """Should raise exception on 500 with detail message."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.json.return_value = {"detail": "AI processing failed"}
            mock_resp.text = "error"
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import upload_prescription
            with pytest.raises(Exception, match="AI processing failed"):
                upload_prescription(b"fake-image", "test.png")

    def test_upload_connection_error(self):
        """Should raise exception on connection failure."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            import requests as real_requests
            mock_requests.post.side_effect = real_requests.exceptions.ConnectionError("Connection refused")
            mock_requests.exceptions = real_requests.exceptions

            from frontend.api_client import upload_prescription
            with pytest.raises(Exception, match="Cannot connect to backend"):
                upload_prescription(b"fake-image", "test.png")


# ─── Analyze Lab Report Tests ───────────────────────────────────────────────


class TestAnalyzeLabReport:
    """Tests for analyze_lab_report function."""

    def test_analyze_success(self):
        """Should return analysis dict on successful analysis."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "report_text": "Hemoglobin: 12.5",
                "analysis": {"patient": {}, "tests": []}
            }
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import analyze_lab_report
            result = analyze_lab_report(b"fake-image", "report.png")
            assert "report_text" in result
            assert "analysis" in result

    def test_analyze_403_raises_error(self):
        """Should raise exception on 403."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import analyze_lab_report
            with pytest.raises(Exception, match="Invalid or missing API key"):
                analyze_lab_report(b"fake-image", "report.png")


# ─── Chat Message Tests ─────────────────────────────────────────────────────


class TestSendChatMessage:
    """Tests for send_chat_message function."""

    def test_chat_success(self):
        """Should return reply string on successful chat."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"reply": "Your hemoglobin is low."}
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import send_chat_message
            result = send_chat_message("What about my hemoglobin?", "Hb: 12.5", "English")
            assert result == "Your hemoglobin is low."

    def test_chat_403_raises_error(self):
        """Should raise exception on 403."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import send_chat_message
            with pytest.raises(Exception, match="Invalid or missing API key"):
                send_chat_message("Hello", "report", "English")


# ─── Diet Recommendation Tests ──────────────────────────────────────────────


class TestDietQuickSuggestions:
    """Tests for get_diet_quick_suggestions function."""

    def test_quick_suggestions_success(self):
        """Should return list of suggestions on success."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "suggestions": ["Eat iron-rich foods", "Avoid fried foods"]
            }
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import get_diet_quick_suggestions
            result = get_diet_quick_suggestions("Cholesterol: 240 (High)")
            assert len(result) == 2
            assert "iron-rich" in result[0]

    def test_quick_suggestions_empty(self):
        """Should return empty list when no suggestions returned."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import get_diet_quick_suggestions
            result = get_diet_quick_suggestions("report text")
            assert result == []


class TestDietPersonalizedPlan:
    """Tests for get_diet_personalized_plan function."""

    def test_personalized_plan_success(self):
        """Should return breakdown and plan on success."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "breakdown": [{"meal": "Breakfast", "calories": 400}],
                "plan": "Eat oats for breakfast."
            }
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import get_diet_personalized_plan
            result = get_diet_personalized_plan(
                "report", "Vegetarian", "General Health", 3, ""
            )
            assert "breakdown" in result
            assert "plan" in result
            assert len(result["breakdown"]) == 1

    def test_personalized_plan_sends_correct_payload(self):
        """Should send correct payload structure to API."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"breakdown": [], "plan": "text"}
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import get_diet_personalized_plan
            get_diet_personalized_plan(
                "Hb: 12.5", "Non-Vegetarian", "Weight Loss", 4, "nuts"
            )

            payload = mock_requests.post.call_args.kwargs["json"]
            assert payload["mode"] == "Personalized Meal Plan"
            assert payload["diet_type"] == "Non-Vegetarian"
            assert payload["goal"] == "Weight Loss"
            assert payload["meals"] == 4
            assert payload["allergies"] == "nuts"

    def test_personalized_plan_empty_allergies(self):
        """Should handle empty allergies gracefully."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"breakdown": [], "plan": "text"}
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import get_diet_personalized_plan
            get_diet_personalized_plan("report", "Vegetarian", "General Health", 3, None)

            payload = mock_requests.post.call_args.kwargs["json"]
            assert payload["allergies"] == ""


# ─── Content-Type Header Handling Tests ─────────────────────────────────────


class TestContentTypeRemoval:
    """Tests that Content-Type is properly removed for file uploads."""

    def test_content_type_removed_for_prescription(self):
        """Content-Type should be removed from headers for file upload."""
        with patch("frontend.api_client.st") as mock_st, \
             patch("frontend.api_client.requests") as mock_requests:
            mock_st.session_state = {"backend_url": "http://test:8000", "api_key": "key"}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"medicines": []}
            mock_requests.post.return_value = mock_resp

            from frontend.api_client import upload_prescription
            upload_prescription(b"fake", "test.png")

            # Check that headers passed to requests.post do NOT have Content-Type
            headers = mock_requests.post.call_args.kwargs["headers"]
            assert "Content-Type" not in headers
            assert "X-API-KEY" in headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
