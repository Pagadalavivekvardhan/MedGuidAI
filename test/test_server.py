"""Unit tests for backend.server FastAPI endpoints."""

import json
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        # Reimport to get fresh app with mocked env
        import importlib
        if "backend.server" in sys.modules:
            del sys.modules["backend.server"]
        from backend.server import app
        return TestClient(app)


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_ok(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestJsonParsingHelpers:
    """Tests for JSON parsing logic used across endpoints."""

    def test_clean_markdown_json(self, parse_json_from_markdown):
        raw_text = '```json\n{"key": "value"}\n```'
        data = parse_json_from_markdown(raw_text)
        assert data["key"] == "value"

    def test_extract_json_array(self, parse_json_from_markdown):
        raw_text = 'Some text\n[{"name": "Test"}]\nMore text'
        data = parse_json_from_markdown(raw_text, "array")
        assert len(data) == 1
        assert data[0]["name"] == "Test"

    def test_set_defaults(self, sample_lab_report_json):
        data = {}
        for key, default in sample_lab_report_json.items():
            data.setdefault(key, default)
        assert "patient" in data
        assert "tests" in data


class TestDietEndpointLogic:
    """Tests for diet recommendation parsing logic."""

    def test_quick_suggestions_extraction(self, sample_suggestions_text):
        suggestions = [line.strip().lstrip("-* ") for line in sample_suggestions_text.split("\n") if line.strip()]
        assert len(suggestions) == 5
        assert "iron-rich foods" in suggestions[0]

    def test_personalized_json_plan_format(self, sample_diet_plan_text, parse_json_from_markdown, sample_diet_json):
        if "JSON:" in sample_diet_plan_text and "PLAN:" in sample_diet_plan_text:
            json_part = sample_diet_plan_text.split("JSON:")[1].split("PLAN:")[0].strip()
            plan_part = sample_diet_plan_text.split("PLAN:")[1].strip()
            breakdown = parse_json_from_markdown(json_part, "array")
            assert len(breakdown) == 4
            assert breakdown[0]["meal"] == "Breakfast"
            assert "breakfast" in plan_part.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
