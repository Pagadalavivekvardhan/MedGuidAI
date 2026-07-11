"""Unit tests for backend.utils.rxnorm module."""

import pytest
from unittest.mock import patch, MagicMock

from backend.utils.rxnorm import (
    correct_medicines_list,
    correct_medicine_name,
    validate_drug_name,
    validate_test_name,
    _fuzzy_correct,
    COMMON_DRUGS,
)


class TestFuzzyCorrect:
    """Tests for _fuzzy_correct function."""

    def test_exact_match_returns_drug(self):
        """Exact match should return the drug name."""
        result = _fuzzy_correct("Paracetamol")
        assert result == "Paracetamol"

    def test_misspelled_drug_corrected(self):
        """Misspelled drug names should be corrected."""
        result = _fuzzy_correct("Paracetmol")  # Missing 'a'
        assert result == "Paracetamol"

    def test_another_misspelling(self):
        """Another common misspelling should be corrected."""
        result = _fuzzy_correct("Iuprofen")  # Missing 'b'
        assert result == "Ibuprofen"

    def test_unknown_drug_returns_none(self):
        """Completely unknown drug should return None."""
        result = _fuzzy_correct("zzzzzzzzzzzzz")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert _fuzzy_correct("") is None

    def test_short_string_returns_none(self):
        """Very short strings (< 3 chars) should return None."""
        assert _fuzzy_correct("ab") is None

    def test_close_match(self):
        """Close matches should be corrected."""
        result = _fuzzy_correct("Paracetamoll")
        assert result == "Paracetamol"


class TestValidateTestName:
    """Tests for validate_test_name function."""

    def test_exact_match(self):
        """Exact match should return the test name."""
        result = validate_test_name("Hemoglobin")
        assert result == "Hemoglobin"

    def test_misspelled_test_corrected(self):
        """Misspelled test names should be corrected."""
        result = validate_test_name("Haemoglobin")  # British spelling
        assert result == "Hemoglobin"

    def test_another_misspelling(self):
        """Another common OCR error should be corrected."""
        result = validate_test_name("Cholestrol")  # Missing 'e'
        assert result == "Total Cholesterol"

    def test_short_name_returns_original(self):
        """Names shorter than 4 chars should be returned as-is."""
        result = validate_test_name("WB")
        assert result == "WB"

    def test_empty_string_returns_original(self):
        """Empty string should be returned as-is."""
        assert validate_test_name("") == ""

    def test_completely_unknown_returns_original(self):
        """Completely unknown test name should be returned as-is."""
        result = validate_test_name("XyzzyTest123")
        assert result == "XyzzyTest123"

    def test_creininine_correction(self):
        """Common OCR error for creatinine should be corrected."""
        result = validate_test_name("Creatinin")  # Missing 'e'
        assert result == "Creatinine"


class TestValidateDrugName:
    """Tests for validate_drug_name function."""

    @patch("backend.utils.rxnorm._validate_rxnorm")
    def test_direct_rxnorm_match(self, mock_validate):
        """Direct RxNorm match should return validated result."""
        mock_validate.return_value = {
            "validated": True,
            "rxcui": "12345",
            "canonical_name": "Aspirin",
            "raw": "Aspirin",
        }
        result = validate_drug_name("Aspirin")
        assert result["validated"] is True
        assert result["canonical_name"] == "Aspirin"
        assert result["rxcui"] == "12345"

    @patch("backend.utils.rxnorm._validate_rxnorm")
    @patch("backend.utils.rxnorm._fuzzy_correct")
    def test_fuzzy_correction_when_direct_fails(self, mock_fuzzy, mock_validate):
        """When direct fails, fuzzy correction should be tried."""
        mock_validate.side_effect = [
            None,  # Direct lookup fails
            {"validated": True, "rxcui": "67890", "canonical_name": "Ibuprofen", "raw": "Ibuprofn"},
        ]
        mock_fuzzy.return_value = "Ibuprofen"

        result = validate_drug_name("Ibuprofn")
        assert result["validated"] is True
        assert result["corrected_from"] == "Ibuprofn"

    @patch("backend.utils.rxnorm._validate_rxnorm")
    @patch("backend.utils.rxnorm._fuzzy_correct")
    def test_unvalidated_drug(self, mock_fuzzy, mock_validate):
        """Drug that can't be validated should return validated=False."""
        mock_validate.return_value = None
        mock_fuzzy.return_value = None

        result = validate_drug_name("UnknownDrugXYZ")
        assert result["validated"] is False
        assert result["raw"] == "UnknownDrugXYZ"


class TestCorrectMedicineName:
    """Tests for correct_medicine_name function."""

    @patch("backend.utils.rxnorm._validate_rxnorm")
    def test_exact_rxnorm_match(self, mock_validate):
        """Exact RxNorm match should return canonical name."""
        mock_validate.return_value = {
            "validated": True,
            "canonical_name": "Aspirin",
            "rxcui": "12345",
            "raw": "Aspirin",
        }
        result = correct_medicine_name("Aspirin")
        assert result == "Aspirin"

    @patch("backend.utils.rxnorm._validate_rxnorm")
    @patch("backend.utils.rxnorm._fuzzy_correct")
    def test_fuzzy_correction(self, mock_fuzzy, mock_validate):
        """Misspelled name should be corrected via fuzzy matching."""
        mock_validate.return_value = None
        mock_fuzzy.return_value = "Paracetamol"

        result = correct_medicine_name("Paracetmol")
        assert result == "Paracetamol"

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        assert correct_medicine_name("") == ""

    def test_whitespace_only(self):
        """Whitespace-only string should be returned as-is."""
        assert correct_medicine_name("   ") == "   "


class TestCorrectMedicinesList:
    """Tests for correct_medicines_list function."""

    @patch("backend.utils.rxnorm.validate_drug_name")
    def test_corrects_validated_drug(self, mock_validate):
        """Validated drug should be corrected with canonical name."""
        mock_validate.return_value = {
            "validated": True,
            "canonical_name": "Aspirin",
            "rxcui": "12345",
            "raw": "Asprin",
        }

        medicines = [{"name": "Asprin", "dosage": "100mg"}]
        result = correct_medicines_list(medicines)

        assert result[0]["name"] == "Aspirin"
        assert result[0]["corrected_from"] == "Asprin"
        assert result[0]["rxnorm_validated"] is True
        assert result[0]["rxcui"] == "12345"

    @patch("backend.utils.rxnorm.validate_drug_name")
    def test_preserves_uncertain_markers(self, mock_validate):
        """Names starting with [ should not be corrected."""
        mock_validate.return_value = None

        medicines = [{"name": "[UNCERTAIN]", "dosage": "unknown"}]
        result = correct_medicines_list(medicines)

        assert result[0]["name"] == "[UNCERTAIN]"
        assert result[0]["rxnorm_validated"] is False

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert correct_medicines_list([]) == []

    @patch("backend.utils.rxnorm.validate_drug_name")
    @patch("backend.utils.rxnorm._fuzzy_correct")
    def test_fuzzy_fallback(self, mock_fuzzy, mock_validate):
        """When RxNorm fails, fuzzy correction should be tried."""
        mock_validate.return_value = {"validated": False, "raw": "Paracetmol"}
        mock_fuzzy.return_value = "Paracetamol"

        medicines = [{"name": "Paracetmol", "dosage": "500mg"}]
        result = correct_medicines_list(medicines)

        assert result[0]["name"] == "Paracetamol"
        assert result[0]["corrected_from"] == "Paracetmol"
        assert result[0]["rxnorm_validated"] is False

    @patch("backend.utils.rxnorm.validate_drug_name")
    def test_unvalidated_drug_preserved(self, mock_validate):
        """Unvalidated drug should keep original name."""
        mock_validate.return_value = {"validated": False, "raw": "UnknownDrug"}
        # Mock fuzzy also returns None
        with patch("backend.utils.rxnorm._fuzzy_correct", return_value=None):
            medicines = [{"name": "UnknownDrug", "dosage": "10mg"}]
            result = correct_medicines_list(medicines)

            assert result[0]["name"] == "UnknownDrug"
            assert result[0]["rxnorm_validated"] is False

    @patch("backend.utils.rxnorm.validate_drug_name")
    def test_multiple_medicines(self, mock_validate):
        """Multiple medicines should all be processed."""
        mock_validate.side_effect = [
            {"validated": True, "canonical_name": "Paracetamol", "rxcui": "111", "raw": "Paracetmol"},
            {"validated": False, "raw": "UnknownDrug"},
            {"validated": True, "canonical_name": "Ibuprofen", "rxcui": "222", "raw": "Iuprofen"},
        ]

        medicines = [
            {"name": "Paracetmol", "dosage": "500mg"},
            {"name": "UnknownDrug", "dosage": "10mg"},
            {"name": "Iuprofen", "dosage": "200mg"},
        ]

        with patch("backend.utils.rxnorm._fuzzy_correct", return_value=None):
            result = correct_medicines_list(medicines)

        assert len(result) == 3
        assert result[0]["name"] == "Paracetamol"
        assert result[1]["name"] == "UnknownDrug"
        assert result[2]["name"] == "Ibuprofen"
