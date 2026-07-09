"""
opencv_service.py
-----------------
Stage 1: OpenCV Image Enhancement Pipeline

Applies a 4-step preprocessing chain to normalize medical document images
before OCR:
  1. Grayscale conversion
  2. Gaussian Blurring (noise suppression)
  3. Adaptive Thresholding (binarization robust to uneven illumination)
  4. Hough Transform deskewing (corrects document tilt)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for a single document image.

    Args:
        image_path: Path to the input image file.

    Returns:
        Preprocessed binary numpy array (grayscale, deskewed).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Step 1 – Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2 – Gaussian Blur to reduce high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3 – Adaptive Thresholding for binarization
    # Uses local neighbourhood to compute threshold, robust to uneven lighting
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # Step 4 – Hough Transform deskewing
    deskewed = _deskew(binary)

    logger.debug("Preprocessing complete for %s", image_path)
    return deskewed


def _deskew(binary_image: np.ndarray) -> np.ndarray:
    """
    Detects document skew angle using Hough Lines and applies
    an affine rotation to correct it.

    Args:
        binary_image: Binary (thresholded) grayscale image.

    Returns:
        Deskewed binary image.
    """
    edges = cv2.Canny(binary_image, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=200)

    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert to degrees relative to horizontal
            a = theta * 180 / np.pi - 90
            angles.append(a)
        angle = float(np.median(angles))
        # Clamp to avoid over-rotation on noisy images
        angle = max(-10.0, min(10.0, angle))

    if abs(angle) < 0.5:
        # Skip rotation for near-horizontal documents
        return binary_image

    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    deskewed = cv2.warpAffine(
        binary_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskewed by %.2f degrees", angle)
    return deskewed


def preprocess_to_pil(image_path: str):
    """
    Preprocesses image and returns a PIL Image object (for Tesseract).
    """
    from PIL import Image
    preprocessed = preprocess_image(image_path)
    return Image.fromarray(preprocessed)
