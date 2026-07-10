"""Unit tests for backend.utils.ocr_engine module."""

import pytest
from unittest.mock import patch

from backend.utils.ocr_engine import (
    extract_text_safe,
    _run_tesseract,
    _run_paddle,
    _arbitrate,
    _make_result,
)


class TestMakeResult:
    """Tests for _make_result helper."""

    def test_returns_correct_structure(self):
        result = _make_result("text", "tesseract", "raw1", "raw2", 85.0)
        assert result["text"] == "text"
        assert result["engine_used"] == "tesseract"
        assert result["tesseract_raw"] == "raw1"
        assert result["paddle_raw"] == "raw2"
        assert result["similarity"] == 85.0


class TestArbitrate:
    """Tests for _arbitrate function."""

    def test_both_empty(self):
        result = _arbitrate("", "")
        assert result["text"] == ""
        assert result["engine_used"] == "none"

    def test_only_tesseract(self):
        result = _arbitrate("hello world", "")
        assert result["text"] == "hello world"
        assert result["engine_used"] == "tesseract"

    def test_only_paddle(self):
        result = _arbitrate("", "hello world")
        assert result["text"] == "hello world"
        assert result["engine_used"] == "paddle"

    def test_high_similarity_prefers_tesseract(self):
        result = _arbitrate("hello world", "hello world")
        assert result["engine_used"] == "tesseract"

    def test_paddle_longer_prefers_paddle(self):
        result = _arbitrate("short", "this is a much longer text from paddle")
        assert result["engine_used"] == "paddle"


class TestRunTesseract:
    """Tests for _run_tesseract function."""

    @patch("backend.utils.ocr_engine.pytesseract", None)
    def test_returns_empty_when_unavailable(self, white_image):
        result = _run_tesseract(white_image)
        assert result == ""


class TestRunPaddle:
    """Tests for _run_paddle function."""

    @patch("backend.utils.ocr_engine._get_paddle", return_value=None)
    def test_returns_empty_when_unavailable(self, _, white_image):
        result = _run_paddle(white_image)
        assert result == ""


class TestExtractTextSafe:
    """Tests for extract_text_safe function."""

    @patch("backend.utils.ocr_engine._get_paddle", return_value=None)
    @patch("backend.utils.ocr_engine.pytesseract", None)
    def test_returns_empty_string(self, _, white_image):
        result = extract_text_safe(white_image)
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
