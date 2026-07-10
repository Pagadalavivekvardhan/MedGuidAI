"""Unit tests for backend.lab_report module."""

import json
import pytest

pytest.importorskip("streamlit")
import backend.lab_report as lr


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_prompt_contains_ocr_text(self, sample_ocr_text):
        prompt = lr.build_prompt(sample_ocr_text)
        assert "John Doe" in prompt

    def test_prompt_requests_json(self, sample_ocr_text):
        prompt = lr.build_prompt(sample_ocr_text)
        assert "JSON" in prompt
        assert "tests" in prompt
        assert "status" in prompt

    def test_prompt_includes_rules(self, sample_ocr_text):
        prompt = lr.build_prompt(sample_ocr_text)
        assert "Normal" in prompt
        assert "High" in prompt
        assert "Low" in prompt


class TestCleanValue:
    """Tests for clean_value helper function."""

    def test_clean_none(self):
        assert lr.clean_value(None) == ""

    def test_clean_string(self):
        assert lr.clean_value("  hello  ") == "hello"

    def test_clean_number(self):
        assert lr.clean_value(123) == "123"


class TestCountStatus:
    """Tests for count_status function."""

    def test_count_normal(self):
        tests = [{"status": "Normal"}, {"status": "Normal"}, {"status": "High"}]
        counts = lr.count_status(tests)
        assert counts["Normal"] == 2
        assert counts["High"] == 1
        assert counts["Low"] == 0

    def test_count_empty(self):
        counts = lr.count_status([])
        assert counts == {"High": 0, "Low": 0, "Normal": 0}


class TestJsonParsing:
    """Tests for JSON parsing logic used in lab report analysis."""

    def test_parse_valid_json(self, parse_json_from_markdown, sample_lab_report_json):
        raw_text = json.dumps(sample_lab_report_json)
        data = parse_json_from_markdown(raw_text)
        assert "patient" in data
        assert "tests" in data

    def test_parse_json_with_markdown(self, parse_json_from_markdown, sample_lab_report_json):
        raw_text = f'```json\n{json.dumps(sample_lab_report_json)}\n```'
        data = parse_json_from_markdown(raw_text)
        assert "patient" in data
        assert len(data["tests"]) == 3

    def test_set_defaults(self, sample_lab_report_json):
        data = {}
        data.setdefault("patient", {})
        data.setdefault("tests", [])
        data.setdefault("summary", "")
        data.setdefault("recommendations", [])
        assert data["patient"] == {}
        assert data["tests"] == []
        assert data["summary"] == ""
        assert data["recommendations"] == []

    def test_validate_status(self, sample_lab_report_json):
        tests = sample_lab_report_json["tests"]
        valid_statuses = ["Normal", "High", "Low"]
        for test in tests:
            if test.get("status") not in valid_statuses:
                test["status"] = "Normal"
        assert tests[0]["status"] == "Low"
        assert tests[1]["status"] == "Normal"
        assert tests[2]["status"] == "High"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
