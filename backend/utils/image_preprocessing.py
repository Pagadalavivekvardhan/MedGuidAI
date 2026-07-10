"""
image_preprocessing.py
---------------------
Enhanced OpenCV Image Preprocessing Pipeline for Medical Documents

Applies advanced preprocessing to normalize medical document images
before OCR:
  1. Grayscale conversion
  2. Denoising (Non-local Means)
  3. Gaussian Blurring (noise suppression)
  4. Adaptive Thresholding (binarization)
  5. Morphological Operations (cleanup)
  6. Hough Transform deskewing (corrects document tilt)
"""

import cv2
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(image_input):
    """
    Full preprocessing pipeline for a medical document image.
    
    Args:
        image_input: Path to image file, numpy array, or PIL Image
    
    Returns:
        Preprocessed binary numpy array (grayscale, deskewed)
    """
    # Handle different input types
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            img = image_input
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    # Step 1 - Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2 - Non-local Means Denoising (excellent for document images)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 3 - Gaussian Blur to reduce high-frequency noise
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Step 4 - Adaptive Thresholding for binarization
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )
    
    # Step 5 - Morphological Operations for cleanup
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 6 - Deskew
    deskewed = _deskew(cleaned)
    
    logger.debug("Preprocessing complete")
    return deskewed


def preprocess_for_ocr(image_input):
    """
    Preprocess image and return a PIL Image object (for Tesseract).
    
    Args:
        image_input: Path to image file, numpy array, or PIL Image
    
    Returns:
        PIL Image object
    """
    preprocessed = preprocess_image(image_input)
    return Image.fromarray(preprocessed)


def enhance_for_vision_model(image: Image.Image) -> Image.Image:
    """
    Minimal contrast enhancement for vision models.
    Vision models like LLaMA-4 work best with original color images,
    so this only applies CLAHE contrast enhancement without binarization.

    Args:
        image: PIL Image (color or grayscale)

    Returns:
        Enhanced PIL Image
    """
    img_array = np.array(image)

    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)

    logger.debug("Enhanced image for vision model")
    return Image.fromarray(enhanced)


def _deskew(binary_image):
    """
    Detects document skew angle using Hough Lines and applies
    an affine rotation to correct it.
    """
    edges = cv2.Canny(binary_image, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=200)
    
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            a = theta * 180 / np.pi - 90
            angles.append(a)
        angle = float(np.median(angles))
        angle = max(-10.0, min(10.0, angle))
    
    if abs(angle) < 0.5:
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
