"""Shared pytest fixtures for MedGuid tests.

Provides reusable fixtures for test images, mock Groq client,
and common test utilities across all test modules.
"""

import pytest
import sys
import os
import json
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Image Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def white_image():
    """Create a plain white RGB test image."""
    return Image.new("RGB", (200, 100), (255, 255, 255))


@pytest.fixture
def text_image():
    """Create a test image with text-like patterns (dark on white)."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    img[40:60, 50:150] = 0  # Dark horizontal band simulating text
    return Image.fromarray(img)


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image."""
    return Image.new("L", (100, 100), 128)


@pytest.fixture
def image_bytes():
    """Create test image as bytes (for UploadFile simulation)."""
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def numpy_image():
    """Create a test image as numpy array."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 200


# ─── Mock Groq Client ───────────────────────────────────────────────────────


@pytest.fixture
def mock_groq_response():
    """Factory fixture to create mock Groq API responses."""

    def _make_response(content: str, model: str = "llama-3.3-70b-versatile"):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    return _make_response


@pytest.fixture
def mock_groq_client(mock_groq_response):
    """Create a mock Groq client with configurable responses."""

    def _make_client(response_text: str):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            response_text
        )
        return mock_client

    return _make_client


# ─── Sample Data Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_prescription_json():
    """Sample prescription JSON response from vision model."""
    return [
        {
            "name": "Paracetamol",
            "dosage": "500mg",
            "frequency": "Three times daily",
            "duration": "5 days",
            "use": "Pain relief and fever reduction",
            "instructions": "Take after food",
        },
        {
            "name": "Amoxicillin",
            "dosage": "250mg",
            "frequency": "Three times daily",
            "duration": "7 days",
            "use": "Antibiotic for infection",
            "instructions": "Complete the full course",
        },
    ]


@pytest.fixture
def sample_lab_report_json():
    """Sample lab report JSON response from Groq."""
    return {
        "patient": {"name": "John Doe", "age": "35", "gender": "Male"},
        "tests": [
            {
                "name": "Hemoglobin",
                "value": "12.5",
                "unit": "g/dL",
                "reference_range": "13.0-17.0",
                "status": "Low",
            },
            {
                "name": "Fasting Glucose",
                "value": "95",
                "unit": "mg/dL",
                "reference_range": "70-100",
                "status": "Normal",
            },
            {
                "name": "Total Cholesterol",
                "value": "240",
                "unit": "mg/dL",
                "reference_range": "125-200",
                "status": "High",
            },
        ],
        "summary": "Hemoglobin is slightly low. Cholesterol is elevated.",
        "recommendations": [
            "Increase iron-rich foods",
            "Reduce saturated fat intake",
            "Follow up in 3 months",
        ],
    }


@pytest.fixture
def sample_diet_json():
    """Sample diet recommendation JSON."""
    return [
        {"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15},
        {"meal": "Lunch", "calories": 600, "carbs": 70, "protein": 35, "fat": 20},
        {"meal": "Dinner", "calories": 500, "carbs": 40, "protein": 40, "fat": 15},
        {"meal": "Snacks", "calories": 200, "carbs": 25, "protein": 5, "fat": 10},
    ]


@pytest.fixture
def sample_ocr_text():
    """Sample OCR extracted text from a lab report."""
    return """
PATIENT: John Doe
AGE: 35
GENDER: Male

COMPLETE BLOOD COUNT
Hemoglobin: 12.5 g/dL (Reference: 13.0-17.0) - Low
WBC: 7500 /uL (Reference: 4000-11000) - Normal
Platelets: 250000 /uL (Reference: 150000-400000) - Normal

LIPID PROFILE
Total Cholesterol: 240 mg/dL (Reference: 125-200) - High
HDL: 45 mg/dL (Reference: >40) - Normal
LDL: 160 mg/dL (Reference: <130) - High
Triglycerides: 180 mg/dL (Reference: <150) - High

FASTING GLUCOSE: 95 mg/dL (Reference: 70-100) - Normal
"""


@pytest.fixture
def sample_diet_plan_text():
    """Sample diet plan response in JSON:PLAN: format."""
    return """JSON:
[
  {"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15},
  {"meal": "Lunch", "calories": 600, "carbs": 70, "protein": 35, "fat": 20},
  {"meal": "Dinner", "calories": 500, "carbs": 40, "protein": 40, "fat": 15},
  {"meal": "Snacks", "calories": 200, "carbs": 25, "protein": 5, "fat": 10}
]
PLAN:
**Breakfast (7:00 AM)**
- Oats with milk and almonds
- 1 boiled egg
- Green tea

**Lunch (12:30 PM)**
- Brown rice with dal
- Grilled chicken breast
- Mixed vegetable salad

**Dinner (7:00 PM)**
- Chapati with paneer curry
- Steamed broccoli
- Buttermilk

**Snacks (4:00 PM)**
- Fruit salad with yogurt
"""


@pytest.fixture
def sample_suggestions_text():
    """Sample quick diet suggestions response."""
    return """- Eat iron-rich foods like spinach, dates, and jaggery to improve hemoglobin
- Include omega-3 rich foods like walnuts and flaxseeds to help lower cholesterol
- Avoid fried and oily foods to reduce triglyceride levels
- Eat small frequent meals instead of heavy meals
- Include whole grains and fiber-rich foods in your daily diet"""


# ─── JSON Parsing Helpers ────────────────────────────────────────────────────


@pytest.fixture
def parse_json_from_markdown():
    """Fixture that returns a JSON parsing helper function."""

    def _parse(raw_text: str, extract_type: str = "object"):
        """Parse JSON from potentially markdown-wrapped text.

        Args:
            raw_text: Raw text that may contain JSON
            extract_type: "object" for {...}, "array" for [...]
        """
        # Clean markdown
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        if extract_type == "array":
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
        else:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1

        if start == -1 or end <= start:
            raise ValueError(f"No JSON {extract_type} found in text")

        return json.loads(raw_text[start:end])

    return _parse
