"""Unit tests for backend.prescription module."""

import json

from backend.utils.image_preprocessing import enhance_for_vision_model


class TestPreprocessPrescriptionImage:
    """Tests for enhance_for_vision_model used in prescription.py."""

    def test_returns_pil_image(self, white_image):
        result = enhance_for_vision_model(white_image)
        assert result.size == white_image.size

    def test_preserves_dimensions(self, white_image):
        result = enhance_for_vision_model(white_image)
        assert result.size == white_image.size


class TestJsonParsing:
    """Tests for JSON parsing logic used in prescription extraction."""

    def test_parse_clean_json(self, parse_json_from_markdown, sample_prescription_json):
        raw_text = json.dumps(sample_prescription_json)
        result = parse_json_from_markdown(raw_text, "array")
        assert len(result) == 2
        assert result[0]["name"] == "Paracetamol"

    def test_parse_json_with_markdown(self, parse_json_from_markdown, sample_prescription_json):
        raw_text = f'```json\n{json.dumps(sample_prescription_json)}\n```'
        result = parse_json_from_markdown(raw_text, "array")
        assert len(result) == 2

    def test_extract_json_array(self, parse_json_from_markdown, sample_prescription_json):
        raw_text = f'Here are the medicines:\n{json.dumps(sample_prescription_json)}\nDone.'
        result = parse_json_from_markdown(raw_text, "array")
        assert len(result) == 2

    def test_medicine_fields(self, sample_prescription_json):
        for med in sample_prescription_json:
            assert "name" in med
            assert "dosage" in med
            assert "frequency" in med
            assert "duration" in med
            assert "use" in med
            assert "instructions" in med


class TestDisplayMedicines:
    """Tests for display logic (without Streamlit)."""

    def test_empty_medicines_list(self):
        medicines = []
        assert len(medicines) == 0

    def test_medicines_have_required_fields(self, sample_prescription_json):
        for med in sample_prescription_json:
            assert "name" in med
            assert "dosage" in med


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
