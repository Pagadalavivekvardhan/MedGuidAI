"""Unit tests for backend.utils.text_correction module."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.utils.text_correction import (
    correct_ocr_text,
    correct_medical_terms,
    correct_drug_name,
    _correct_common_errors,
    MEDICAL_TERMS,
    DRUG_NAMES,
)


class TestCorrectOcrText:
    """Tests for the main correct_ocr_text function."""

    def test_empty_string(self):
        assert correct_ocr_text("") == ""

    def test_none_returns_none(self):
        assert correct_ocr_text(None) is None

    def test_whitespace_normalization(self):
        result = correct_ocr_text("hello   world\t\ttest")
        assert "  " not in result
        assert "\t" not in result

    def test_newline_normalization(self):
        result = correct_ocr_text("line1\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_common_misspelling_fix(self):
        result = correct_ocr_text("The blod sample was analyzed")
        assert "blood" in result

    def test_medical_term_correction(self):
        result = correct_ocr_text("Hemoglobin level is normal")
        assert "hemoglobin" in result.lower()

    def test_preserves_original_whitespace_structure(self):
        result = correct_ocr_text("test value: 12.5\nreference: 10-15")
        assert "reference" in result


class TestCorrectMedicalTerms:
    """Tests for medical term correction."""

    def test_known_medical_term(self):
        result = correct_medical_terms("hemoglobin")
        assert result == "hemoglobin"

    def test_misspelled_medical_term(self):
        result = correct_medical_terms("hemglobin")
        assert result == "hemoglobin"

    def test_preserves_capitalization(self):
        result = correct_medical_terms("Hemoglobin")
        assert result[0].isupper()

    def test_short_words_not_corrected(self):
        # Words shorter than 3 characters should not be corrected
        result = correct_medical_terms("mg")
        assert result == "mg"

    def test_no_match_returns_original(self):
        result = correct_medical_terms("xyzabc")
        assert result == "xyzabc"

    def test_multiple_words(self):
        result = correct_medical_terms("The hemoglobin level")
        assert "hemoglobin" in result.lower()


class TestCorrectDrugName:
    """Tests for drug name correction."""

    def test_known_drug_name(self):
        result = correct_drug_name("Paracetamol")
        assert result == "Paracetamol"

    def test_misspelled_drug_name(self):
        result = correct_drug_name("Paracetaml")
        assert result is not None  # Should find a close match

    def test_unknown_drug_returns_none(self):
        result = correct_drug_name("xyzabc123")
        assert result is None

    def test_empty_string(self):
        result = correct_drug_name("")
        assert result is None


class TestCorrectCommonErrors:
    """Tests for common OCR error correction."""

    def test_blod_to_blood(self):
        result = _correct_common_errors("blod sample")
        assert "blood" in result

    def test_platlet_to_platelet(self):
        result = _correct_common_errors("platlet count")
        assert "platelet" in result

    def test_case_insensitive(self):
        result = _correct_common_errors("BLOD sample")
        assert "blood" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
