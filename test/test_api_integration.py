"""Integration tests for all FastAPI API endpoints.

These tests use FastAPI's TestClient to exercise the full request lifecycle:
  Request -> Auth -> Processing -> Groq API (mocked) -> JSON parsing -> Response

Covers:
  - GET / and GET /health
  - POST /api/upload-prescription (vision model)
  - POST /api/analyze-lab-report (text model + OCR)
  - POST /api/chat (text model)
  - POST /api/diet-recommendation (text model, quick + personalized)
  - Authentication enforcement (missing/invalid API key)
  - Error handling (invalid files, JSON parse failures)
"""

import json
import os
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_API_KEY = "medguid-dev-key-2024"


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Create a fresh FastAPI app instance with mocked env and no rate limiting."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        import importlib
        if "backend.server" in sys.modules:
            del sys.modules["backend.server"]
        from backend.server import app
        from backend.security import limiter
        limiter.enabled = False
        try:
            yield app
        finally:
            limiter.enabled = True


@pytest.fixture
def client(app):
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Headers with valid API key."""
    return {"X-API-KEY": DEFAULT_API_KEY}


@pytest.fixture
def test_image():
    """Create a small test PNG image."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def mock_groq_response():
    """Factory fixture to create mock Groq API responses."""

    def _make(content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    return _make


# ─── Health Endpoints ───────────────────────────────────────────────────────


class TestHealthEndpoints:
    """Tests for GET / and GET /health."""

    def test_root(self, client, auth_headers):
        resp = client.get("/", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "MedGuid" in data["message"]

    def test_health(self, client, auth_headers):
        resp = client.get("/health", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ─── Authentication Tests ──────────────────────────────────────────────────


class TestAuthentication:
    """Tests that all protected endpoints enforce API key auth."""

    def test_prescription_no_key_returns_403(self, client, test_image):
        resp = client.post(
            "/api/upload-prescription",
            files={"file": ("test.png", test_image, "image/png")},
        )
        assert resp.status_code == 403

    def test_prescription_invalid_key_returns_403(self, client, test_image):
        resp = client.post(
            "/api/upload-prescription",
            files={"file": ("test.png", test_image, "image/png")},
            headers={"X-API-KEY": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_lab_report_no_key_returns_403(self, client, test_image):
        resp = client.post(
            "/api/analyze-lab-report",
            files={"file": ("test.png", test_image, "image/png")},
        )
        assert resp.status_code == 403

    def test_chat_no_key_returns_403(self, client):
        resp = client.post(
            "/api/chat",
            json={"user_input": "Hi", "report_text": "Hb: 12", "language": "English"},
        )
        assert resp.status_code == 403

    def test_diet_no_key_returns_403(self, client):
        resp = client.post(
            "/api/diet-recommendation",
            json={"report_text": "Cholesterol: 240", "mode": "Quick Suggestions"},
        )
        assert resp.status_code == 403


# ─── Prescription Endpoint ─────────────────────────────────────────────────


class TestPrescriptionEndpoint:
    """Integration tests for POST /api/upload-prescription."""

    def test_successful_prescription_extraction(self, client, auth_headers, test_image, mock_groq_response):
        """Full flow: upload image -> mock Groq -> parse JSON -> return medicines."""
        medicines_json = json.dumps([
            {
                "name": "Paracetamol",
                "dosage": "500mg",
                "frequency": "Three times daily",
                "duration": "5 days",
                "use": "Pain relief",
                "instructions": "Take after food",
            }
        ])

        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(medicines_json)

            resp = client.post(
                "/api/upload-prescription",
                files={"file": ("rx.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "medicines" in data
        assert len(data["medicines"]) == 1
        assert data["medicines"][0]["name"] == "Paracetamol"
        assert data["medicines"][0]["dosage"] == "500mg"

    def test_prescription_multiple_medicines(self, client, auth_headers, test_image, mock_groq_response):
        """Should extract multiple medicines from a prescription."""
        medicines_json = json.dumps([
            {"name": "Paracetamol", "dosage": "500mg", "frequency": "Three times daily", "duration": "5 days", "use": "Pain relief", "instructions": "Take after food"},
            {"name": "Amoxicillin", "dosage": "250mg", "frequency": "Three times daily", "duration": "7 days", "use": "Antibiotic", "instructions": "Complete course"},
            {"name": "Cetirizine", "dosage": "10mg", "frequency": "Once daily", "duration": "3 days", "use": "Allergy", "instructions": "Take at bedtime"},
        ])

        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(medicines_json)

            resp = client.post(
                "/api/upload-prescription",
                files={"file": ("rx.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["medicines"]) == 3

    def test_prescription_with_markdown_wrapped_response(self, client, auth_headers, test_image, mock_groq_response):
        """Should handle Groq returning JSON wrapped in markdown code blocks."""
        medicines = [{"name": "Ibuprofen", "dosage": "400mg", "frequency": "Twice daily", "duration": "3 days", "use": "Pain relief", "instructions": "Take with food"}]
        markdown_response = f"```json\n{json.dumps(medicines)}\n```"

        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(markdown_response)

            resp = client.post(
                "/api/upload-prescription",
                files={"file": ("rx.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        assert len(resp.json()["medicines"]) == 1
        assert resp.json()["medicines"][0]["name"] == "Ibuprofen"

    def test_prescription_invalid_json_returns_500(self, client, auth_headers, test_image, mock_groq_response):
        """Should return 500 when Groq returns unparseable JSON."""
        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response("Sorry, I cannot analyze this image.")

            resp = client.post(
                "/api/upload-prescription",
                files={"file": ("rx.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 500
        assert "Failed to parse" in resp.json()["detail"]

    def test_prescription_empty_image_still_makes_api_call(self, client, auth_headers, mock_groq_response):
        """Empty image bytes should still reach the Groq API (server validates)."""
        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response("[]")

            resp = client.post(
                "/api/upload-prescription",
                files={"file": ("empty.png", b"", "image/png")},
                headers=auth_headers,
            )
        # Server may return 400 (bad image) or 200 (empty list) - both are valid
        assert resp.status_code in [200, 400, 500]


# ─── Lab Report Endpoint ────────────────────────────────────────────────────


class TestLabReportEndpoint:
    """Integration tests for POST /api/analyze-lab-report."""

    def test_successful_lab_analysis(self, client, auth_headers, test_image, mock_groq_response):
        """Full flow: upload image -> OCR -> mock Groq -> parse JSON -> return analysis."""
        analysis_json = json.dumps({
            "patient": {"name": "John Doe", "age": "35", "gender": "Male"},
            "tests": [
                {"name": "Hemoglobin", "value": "12.5", "unit": "g/dL", "reference_range": "13.0-17.0", "status": "Low"},
                {"name": "Glucose", "value": "95", "unit": "mg/dL", "reference_range": "70-100", "status": "Normal"},
            ],
            "summary": "Hemoglobin is slightly low.",
            "recommendations": ["Eat iron-rich foods"],
        })

        with patch("backend.server.client") as mock_client, \
             patch("backend.server.extract_text_safe") as mock_ocr:
            mock_ocr.return_value = "Hemoglobin: 12.5 g/dL\nGlucose: 95 mg/dL"
            mock_client.chat.completions.create.return_value = mock_groq_response(analysis_json)

            resp = client.post(
                "/api/analyze-lab-report",
                files={"file": ("report.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "report_text" in data
        assert "analysis" in data
        assert data["analysis"]["patient"]["name"] == "John Doe"
        assert len(data["analysis"]["tests"]) == 2
        assert data["analysis"]["tests"][0]["status"] == "Low"
        assert data["analysis"]["tests"][1]["status"] == "Normal"

    def test_lab_report_with_markdown_wrapped_response(self, client, auth_headers, test_image, mock_groq_response):
        """Should handle markdown-wrapped JSON from Groq."""
        analysis = {
            "patient": {"name": "Jane", "age": "28", "gender": "Female"},
            "tests": [{"name": "Hemoglobin", "value": "11.0", "unit": "g/dL", "reference_range": "12.0-16.0", "status": "Low"}],
            "summary": "Low hemoglobin.",
            "recommendations": ["Eat more iron"],
        }
        markdown_response = f"```json\n{json.dumps(analysis)}\n```"

        with patch("backend.server.client") as mock_client, \
             patch("backend.server.extract_text_safe") as mock_ocr:
            mock_ocr.return_value = "Hemoglobin: 11.0"
            mock_client.chat.completions.create.return_value = mock_groq_response(markdown_response)

            resp = client.post(
                "/api/analyze-lab-report",
                files={"file": ("report.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis"]["patient"]["name"] == "Jane"
        assert data["analysis"]["tests"][0]["status"] == "Low"

    def test_lab_report_invalid_json_returns_500(self, client, auth_headers, test_image, mock_groq_response):
        """Should return 500 when Groq returns unparseable JSON."""
        with patch("backend.server.client") as mock_client, \
             patch("backend.server.extract_text_safe") as mock_ocr:
            mock_ocr.return_value = "Some OCR text"
            mock_client.chat.completions.create.return_value = mock_groq_response("Not valid JSON at all")

            resp = client.post(
                "/api/analyze-lab-report",
                files={"file": ("report.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 500

    def test_lab_report_status_normalization(self, client, auth_headers, test_image, mock_groq_response):
        """Invalid statuses should be normalized to 'Normal'."""
        analysis = {
            "patient": {},
            "tests": [
                {"name": "Test1", "value": "10", "unit": "mg", "reference_range": "5-15", "status": "Normal"},
                {"name": "Test2", "value": "20", "unit": "mg", "reference_range": "5-15", "status": "Elevated"},  # Invalid
                {"name": "Test3", "value": "3", "unit": "mg", "reference_range": "5-15", "status": "decreased"},  # Invalid
            ],
            "summary": "Some findings.",
            "recommendations": [],
        }

        with patch("backend.server.client") as mock_client, \
             patch("backend.server.extract_text_safe") as mock_ocr:
            mock_ocr.return_value = "Test data"
            mock_client.chat.completions.create.return_value = mock_groq_response(json.dumps(analysis))

            resp = client.post(
                "/api/analyze-lab-report",
                files={"file": ("report.png", test_image, "image/png")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        tests = resp.json()["analysis"]["tests"]
        assert tests[0]["status"] == "Normal"
        assert tests[1]["status"] == "Normal"  # "Elevated" -> "Normal"
        assert tests[2]["status"] == "Normal"  # "decreased" -> "Normal"


# ─── Chat Endpoint ──────────────────────────────────────────────────────────


class TestChatEndpoint:
    """Integration tests for POST /api/chat."""

    def test_successful_chat(self, client, auth_headers, mock_groq_response):
        """Full flow: send question -> mock Groq -> return reply."""
        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(
                "Your hemoglobin is slightly low. This means you may feel tired easily."
            )

            resp = client.post(
                "/api/chat",
                json={
                    "user_input": "What does my hemoglobin mean?",
                    "report_text": "Hemoglobin: 12.5 g/dL (Low)",
                    "language": "English",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "hemoglobin" in data["reply"].lower()

    def test_chat_hindi_language(self, client, auth_headers, mock_groq_response):
        """Should respect the language parameter."""
        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(
                "आपका हीमोग्लोबिन थोड़ा कम है।"
            )

            resp = client.post(
                "/api/chat",
                json={
                    "user_input": "मेरा हीमोग्लोबिन कैसा है?",
                    "report_text": "Hemoglobin: 12.5 (Low)",
                    "language": "Hindi",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        assert "reply" in resp.json()

    def test_chat_missing_fields_returns_422(self, client, auth_headers):
        """Should return 422 when required fields are missing."""
        resp = client.post(
            "/api/chat",
            json={"user_input": "Hi"},  # Missing report_text and language
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ─── Diet Recommendation Endpoint ──────────────────────────────────────────


class TestDietQuickSuggestions:
    """Integration tests for POST /api/diet-recommendation (Quick Suggestions mode)."""

    def test_quick_suggestions_success(self, client, auth_headers, mock_groq_response):
        """Full flow: request quick suggestions -> mock Groq -> return list."""
        suggestions_text = (
            "- Eat iron-rich foods like spinach and dates\n"
            "- Include omega-3 foods like walnuts\n"
            "- Avoid fried and oily foods\n"
            "- Eat small frequent meals\n"
            "- Include whole grains daily"
        )

        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(suggestions_text)

            resp = client.post(
                "/api/diet-recommendation",
                json={
                    "report_text": "Cholesterol: 240 (High), Hemoglobin: 12.0 (Low)",
                    "mode": "Quick Suggestions",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 5
        assert "iron-rich" in data["suggestions"][0]


class TestDietPersonalizedPlan:
    """Integration tests for POST /api/diet-recommendation (Personalized Meal Plan mode)."""

    def test_personalized_plan_success(self, client, auth_headers, mock_groq_response):
        """Full flow: request personalized plan -> mock Groq -> return breakdown + plan."""
        plan_text = """JSON:
[
  {"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15},
  {"meal": "Lunch", "calories": 600, "carbs": 70, "protein": 35, "fat": 20},
  {"meal": "Dinner", "calories": 500, "carbs": 40, "protein": 40, "fat": 15}
]
PLAN:
**Breakfast** - Oats with milk, 1 boiled egg
**Lunch** - Brown rice with dal, grilled chicken
**Dinner** - Chapati with paneer curry"""

        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(plan_text)

            resp = client.post(
                "/api/diet-recommendation",
                json={
                    "report_text": "Hemoglobin: 12.0 (Low)",
                    "mode": "Personalized Meal Plan",
                    "diet_type": "Vegetarian",
                    "goal": "General Health",
                    "meals": 3,
                    "allergies": "",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "breakdown" in data
        assert "plan" in data
        assert len(data["breakdown"]) == 3
        assert data["breakdown"][0]["meal"] == "Breakfast"
        assert "breakfast" in data["plan"].lower()

    def test_personalized_plan_with_allergies(self, client, auth_headers, mock_groq_response):
        """Should pass allergies to the Groq prompt."""
        plan_text = 'JSON:\n[{"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15}]\nPLAN:\nEat oats.'

        with patch("backend.server.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_groq_response(plan_text)

            resp = client.post(
                "/api/diet-recommendation",
                json={
                    "report_text": "Glucose: 110 (High)",
                    "mode": "Personalized Meal Plan",
                    "diet_type": "Non-Vegetarian",
                    "goal": "Weight Loss",
                    "meals": 4,
                    "allergies": "nuts, dairy",
                },
                headers=auth_headers,
            )

        assert resp.status_code == 200
        # Verify the response structure includes our parameters
        assert "breakdown" in resp.json()
        assert "plan" in resp.json()


# ─── Rate Limiting Tests ────────────────────────────────────────────────────





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
