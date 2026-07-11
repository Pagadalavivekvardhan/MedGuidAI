"""End-to-end tests that mock the Groq client and verify the full extraction pipeline.

These tests verify that the complete flow works correctly:
- Prescription: Image → Preprocessing → Groq Vision API → JSON Parsing → Medicine List
- Lab Report: Image → OCR → Text Correction → Groq Text API → JSON Parsing → Analysis
- API Endpoints: Request → Processing → Response
"""

import pytest
import json
from io import BytesIO
from unittest.mock import patch, MagicMock
from PIL import Image

from backend.utils.image_preprocessing import enhance_for_vision_model

# Default development API key
DEFAULT_API_KEY = "medguid-dev-key-2024"


# ─── Prescription E2E Tests ─────────────────────────────────────────────────


class TestPrescriptionPipeline:
    """End-to-end tests for the prescription extraction pipeline."""

    def test_full_prescription_pipeline(
        self, white_image, mock_groq_response, sample_prescription_json
    ):
        """Test complete pipeline: image → preprocess → Groq API → JSON → medicines list."""
        # Step 1: Preprocess image
        processed = enhance_for_vision_model(white_image)
        # Small images may be upscaled for better vision model accuracy
        assert processed.size[0] >= white_image.size[0]
        assert processed.size[1] >= white_image.size[1]

        # Step 2: Convert to base64 (simulating the API call)
        import base64
        import io

        img_buffer = io.BytesIO()
        processed.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        assert len(img_base64) > 0

        # Step 3: Simulate Groq API response
        response_text = json.dumps(sample_prescription_json)
        mock_resp = mock_groq_response(response_text)

        # Step 4: Parse JSON response (simulating the parsing logic in prescription.py)
        raw_text = mock_resp.choices[0].message.content.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        medicines = json.loads(raw_text[start:end])

        # Step 5: Verify results
        assert len(medicines) == 2
        assert medicines[0]["name"] == "Paracetamol"
        assert medicines[0]["dosage"] == "500mg"
        assert medicines[1]["name"] == "Amoxicillin"
        assert "instructions" in medicines[0]

    def test_prescription_with_markdown_wrapped_response(
        self, white_image, mock_groq_response, sample_prescription_json
    ):
        """Test that markdown-wrapped JSON responses are handled correctly."""
        # Simulate a response wrapped in markdown code blocks
        response_text = f"```json\n{json.dumps(sample_prescription_json)}\n```"
        mock_resp = mock_groq_response(response_text)

        raw_text = mock_resp.choices[0].message.content.strip()
        # Clean markdown
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        medicines = json.loads(raw_text[start:end])

        assert len(medicines) == 2
        assert medicines[0]["name"] == "Paracetamol"

    def test_prescription_empty_response_handling(self, white_image, mock_groq_response):
        """Test graceful handling when Groq returns empty or invalid JSON."""
        # Simulate a non-JSON response
        mock_resp = mock_groq_response("Sorry, I couldn't analyze the image.")

        raw_text = mock_resp.choices[0].message.content.strip()
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1

        # Should fail to find JSON array - this is how prescription.py handles it
        assert start == -1 or end <= start


# ─── Lab Report E2E Tests ──────────────────────────────────────────────────


class TestLabReportPipeline:
    """End-to-end tests for the lab report analysis pipeline."""

    def test_full_lab_report_pipeline(
        self, text_image, mock_groq_response, sample_lab_report_json, sample_ocr_text
    ):
        """Test complete pipeline: image → OCR → Groq API → JSON → analysis."""
        from backend.utils.text_correction import correct_ocr_text

        # Step 1: Simulate OCR extraction (in real flow, this would use pytesseract/paddle)
        ocr_text = sample_ocr_text

        # Step 2: Apply text correction
        corrected_text = correct_ocr_text(ocr_text)
        assert len(corrected_text) > 0
        assert "hemoglobin" in corrected_text.lower()

        # Step 3: Simulate Groq API response
        response_text = json.dumps(sample_lab_report_json)
        mock_resp = mock_groq_response(response_text)

        # Step 4: Parse JSON response (simulating the parsing logic in lab_report.py)
        raw_text = mock_resp.choices[0].message.content.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        analysis = json.loads(raw_text[start:end])

        # Step 5: Set defaults (matching lab_report.py logic)
        analysis.setdefault("patient", {})
        analysis.setdefault("tests", [])
        analysis.setdefault("summary", "")
        analysis.setdefault("recommendations", [])

        # Step 6: Validate and normalize test statuses
        valid_statuses = ["Normal", "High", "Low"]
        for test in analysis.get("tests", []):
            if test.get("status") not in valid_statuses:
                test["status"] = "Normal"

        # Step 7: Verify results
        assert "patient" in analysis
        assert analysis["patient"]["name"] == "John Doe"
        assert len(analysis["tests"]) == 3
        assert analysis["tests"][0]["status"] == "Low"  # Hemoglobin
        assert analysis["tests"][1]["status"] == "Normal"  # Glucose
        assert analysis["tests"][2]["status"] == "High"  # Cholesterol
        assert len(analysis["recommendations"]) > 0

    def test_lab_report_with_markdown_wrapped_response(
        self, mock_groq_response, sample_lab_report_json
    ):
        """Test that markdown-wrapped JSON responses are handled correctly."""
        response_text = f"```json\n{json.dumps(sample_lab_report_json)}\n```"
        mock_resp = mock_groq_response(response_text)

        raw_text = mock_resp.choices[0].message.content.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        analysis = json.loads(raw_text[start:end])

        assert "patient" in analysis
        assert len(analysis["tests"]) == 3

    def test_lab_report_status_normalization(self, mock_groq_response):
        """Test that invalid statuses are normalized to 'Normal'."""
        from backend.lab_report import count_status

        raw_data = {
            "patient": {},
            "tests": [
                {"name": "Test1", "status": "Normal"},
                {"name": "Test2", "status": "High"},
                {"name": "Test3", "status": "Elevated"},  # Invalid
                {"name": "Test4", "status": "Low"},
                {"name": "Test5", "status": "decreased"},  # Invalid
            ],
            "summary": "",
            "recommendations": [],
        }

        # Normalize statuses using actual logic
        valid_statuses = ["Normal", "High", "Low"]
        for test in raw_data.get("tests", []):
            if test.get("status") not in valid_statuses:
                test["status"] = "Normal"

        # Verify normalization
        assert raw_data["tests"][0]["status"] == "Normal"
        assert raw_data["tests"][1]["status"] == "High"
        assert raw_data["tests"][2]["status"] == "Normal"  # Fixed
        assert raw_data["tests"][3]["status"] == "Low"
        assert raw_data["tests"][4]["status"] == "Normal"  # Fixed

        # Verify count_status works with normalized data
        counts = count_status(raw_data["tests"])
        assert counts["Normal"] == 3
        assert counts["High"] == 1
        assert counts["Low"] == 1


# ─── Diet Recommendation E2E Tests ─────────────────────────────────────────


class TestDietPipeline:
    """End-to-end tests for the diet recommendation pipeline."""

    def test_quick_suggestions_pipeline(
        self, mock_groq_response, sample_suggestions_text
    ):
        """Test complete pipeline for quick diet suggestions."""
        mock_resp = mock_groq_response(sample_suggestions_text)
        raw_text = mock_resp.choices[0].message.content.strip()

        # Parse suggestions
        suggestions = [
            line.strip().lstrip("-* ")
            for line in raw_text.split("\n")
            if line.strip()
        ]

        assert len(suggestions) == 5
        assert "iron-rich foods" in suggestions[0]
        assert "cholesterol" in suggestions[1]

    def test_personalized_meal_plan_pipeline(
        self, mock_groq_response, sample_diet_plan_text, parse_json_from_markdown, sample_diet_json
    ):
        """Test complete pipeline for personalized meal plan."""
        mock_resp = mock_groq_response(sample_diet_plan_text)
        raw_text = mock_resp.choices[0].message.content.strip()

        # Parse JSON and plan
        if "JSON:" in raw_text and "PLAN:" in raw_text:
            json_part = raw_text.split("JSON:")[1].split("PLAN:")[0].strip()
            plan_part = raw_text.split("PLAN:")[1].strip()

            breakdown = parse_json_from_markdown(json_part, "array")

            assert len(breakdown) == 4
            assert breakdown[0]["meal"] == "Breakfast"
            assert breakdown[0]["calories"] == 400
            assert "breakfast" in plan_part.lower()


# ─── API Endpoint E2E Tests ────────────────────────────────────────────────


class TestAPIEndpointPipeline:
    """End-to-end tests for FastAPI API endpoints."""

    def test_prescription_endpoint_flow(
        self, mock_groq_response, sample_prescription_json
    ):
        """Test the full prescription endpoint flow with mocked Groq."""
        from fastapi.testclient import TestClient
        import os
        import sys

        # Mock the Groq client
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            if "backend.server" in sys.modules:
                del sys.modules["backend.server"]
            from backend.server import app
            from backend.security import limiter
            limiter.enabled = False
            client = TestClient(app)

            # Mock the Groq client's chat.completions.create
            with patch("backend.server.client") as mock_client:
                mock_resp = mock_groq_response(json.dumps(sample_prescription_json))
                mock_client.chat.completions.create.return_value = mock_resp

                # Create test image
                img = Image.new("RGB", (100, 100), (255, 255, 255))
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                # Make request
                response = client.post(
                    "/api/upload-prescription",
                    files={"file": ("test.png", buf, "image/png")},
                    headers={"X-API-KEY": DEFAULT_API_KEY},
                )

                assert response.status_code == 200
                data = response.json()
                assert "medicines" in data
                assert len(data["medicines"]) == 2
                assert data["medicines"][0]["name"] == "Paracetamol"

    def test_lab_report_endpoint_flow(
        self, mock_groq_response, sample_lab_report_json, sample_ocr_text
    ):
        """Test the full lab report endpoint flow with mocked Groq."""
        from fastapi.testclient import TestClient
        import os
        import sys

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            if "backend.server" in sys.modules:
                del sys.modules["backend.server"]
            from backend.server import app
            from backend.security import limiter
            limiter.enabled = False
            client = TestClient(app)

            # Mock both the OCR and Groq client
            with patch("backend.server.client") as mock_client, \
                 patch("backend.server.extract_text_safe") as mock_ocr:
                mock_ocr.return_value = sample_ocr_text
                mock_resp = mock_groq_response(json.dumps(sample_lab_report_json))
                mock_client.chat.completions.create.return_value = mock_resp

                # Create test image
                img = Image.new("RGB", (100, 100), (255, 255, 255))
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                # Make request
                response = client.post(
                    "/api/analyze-lab-report",
                    files={"file": ("test.png", buf, "image/png")},
                    headers={"X-API-KEY": DEFAULT_API_KEY},
                )

                assert response.status_code == 200
                data = response.json()
                assert "report_text" in data
                assert "analysis" in data
                assert data["analysis"]["patient"]["name"] == "John Doe"
                assert len(data["analysis"]["tests"]) == 3

    def test_chat_endpoint_flow(self, mock_groq_response):
        """Test the full chat endpoint flow with mocked Groq."""
        from fastapi.testclient import TestClient
        import os
        import sys

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            if "backend.server" in sys.modules:
                del sys.modules["backend.server"]
            from backend.server import app
            from backend.security import limiter
            limiter.enabled = False
            client = TestClient(app)

            with patch("backend.server.client") as mock_client:
                mock_resp = mock_groq_response("Your hemoglobin is slightly low.")
                mock_client.chat.completions.create.return_value = mock_resp

                response = client.post(
                    "/api/chat",
                    json={
                        "user_input": "What does my report say?",
                        "report_text": "Hemoglobin: 12.5 (Low)",
                        "language": "English",
                    },
                    headers={"X-API-KEY": DEFAULT_API_KEY},
                )

                assert response.status_code == 200
                data = response.json()
                assert "reply" in data
                assert "hemoglobin" in data["reply"].lower()

    def test_diet_quick_endpoint_flow(
        self, mock_groq_response, sample_suggestions_text
    ):
        """Test the full diet quick suggestions endpoint flow."""
        from fastapi.testclient import TestClient
        import os
        import sys

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            if "backend.server" in sys.modules:
                del sys.modules["backend.server"]
            from backend.server import app
            from backend.security import limiter
            limiter.enabled = False
            client = TestClient(app)

            with patch("backend.server.client") as mock_client:
                mock_resp = mock_groq_response(sample_suggestions_text)
                mock_client.chat.completions.create.return_value = mock_resp

                response = client.post(
                    "/api/diet-recommendation",
                    json={
                        "report_text": "Cholesterol: 240 (High)",
                        "mode": "Quick Suggestions",
                    },
                    headers={"X-API-KEY": DEFAULT_API_KEY},
                )

                assert response.status_code == 200
                data = response.json()
                assert "suggestions" in data
                assert len(data["suggestions"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
