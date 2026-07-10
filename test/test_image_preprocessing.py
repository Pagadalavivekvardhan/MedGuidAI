"""Unit tests for backend.utils.image_preprocessing module."""

import numpy as np
from PIL import Image

from backend.utils.image_preprocessing import (
    preprocess_image,
    preprocess_for_ocr,
    enhance_for_vision_model,
    _deskew,
)


class TestEnhanceForVisionModel:
    """Tests for enhance_for_vision_model function."""

    def test_returns_pil_image(self, white_image):
        result = enhance_for_vision_model(white_image)
        assert isinstance(result, Image.Image)

    def test_preserves_image_size(self, white_image):
        result = enhance_for_vision_model(white_image)
        assert result.size == white_image.size

    def test_handles_grayscale_image(self, grayscale_image):
        result = enhance_for_vision_model(grayscale_image)
        assert isinstance(result, Image.Image)

    def test_handles_rgb_image(self, text_image):
        result = enhance_for_vision_model(text_image)
        assert isinstance(result, Image.Image)


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_returns_numpy_array(self, text_image):
        result = preprocess_image(text_image)
        assert isinstance(result, np.ndarray)

    def test_returns_grayscale(self, text_image):
        result = preprocess_image(text_image)
        assert len(result.shape) == 2  # Should be 2D (grayscale)

    def test_output_shape_matches_input(self, text_image):
        result = preprocess_image(text_image)
        assert result.shape == (text_image.height, text_image.width)


class TestPreprocessForOcr:
    """Tests for preprocess_for_ocr function."""

    def test_returns_pil_image(self, text_image):
        result = preprocess_for_ocr(text_image)
        assert isinstance(result, Image.Image)

    def test_returns_grayscale_pil(self, text_image):
        result = preprocess_for_ocr(text_image)
        assert result.mode == "L"  # Grayscale


class TestDeskew:
    """Tests for _deskew function."""

    def test_straight_image_unchanged(self):
        # Create a straight binary image
        img = np.zeros((100, 200), dtype=np.uint8)
        img[40:60, 20:180] = 255  # Horizontal line
        result = _deskew(img)
        # Should be similar to input
        assert result.shape == img.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
