"""
ocr_engine.py
-------------
Dual-Engine OCR System

Runs PaddleOCR and Tesseract in parallel on a preprocessed image.
A heuristic arbitration layer compares both outputs:
  - If similarity >= threshold -> accept Tesseract output (more structured)
  - If PaddleOCR is significantly longer -> prefer PaddleOCR
  - Otherwise -> use the longer output

Returns the best-quality text string for downstream processing.
"""

import logging
import platform
import numpy as np
from PIL import Image
from .image_preprocessing import preprocess_for_ocr

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

logger = logging.getLogger(__name__)

# Configure Tesseract path
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Lazy-load PaddleOCR to avoid slow import at startup
_paddle_ocr = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except ImportError:
            logger.warning("PaddleOCR not available")
            return None
    return _paddle_ocr


def extract_text_dual(image_input):
    """
    Full dual-engine OCR pipeline for a document image.
    
    Args:
        image_input: Path to image file, numpy array, or PIL Image
    
    Returns:
        dict with keys:
          - text: final chosen OCR text
          - engine_used: "tesseract" | "paddle" | "combined"
          - tesseract_raw: raw Tesseract output
          - paddle_raw: raw PaddleOCR output
          - similarity: fuzz.ratio score between both outputs
    """
    # Run both engines
    tess_text = _run_tesseract(image_input)
    paddle_text = _run_paddle(image_input)
    
    # Arbitration
    result = _arbitrate(tess_text, paddle_text)
    logger.info(
        "OCR complete. Engine: %s | Similarity: %.1f",
        result["engine_used"],
        result["similarity"],
    )
    return result


def extract_text(image_input):
    """
    Simple text extraction using the best available engine.
    Returns just the text string.
    """
    result = extract_text_dual(image_input)
    return result["text"]


def _run_tesseract(image_input):
    """Run Tesseract OCR with page segmentation mode 6."""
    if pytesseract is None:
        logger.warning("Tesseract not available")
        return ""
    try:
        if isinstance(image_input, np.ndarray):
            pil_image = Image.fromarray(image_input)
        elif isinstance(image_input, str):
            pil_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            pil_image = Image.fromarray(np.array(image_input))
        
        text = pytesseract.image_to_string(pil_image, config="--psm 6")
        return text.strip()
    except Exception as exc:
        logger.warning("Tesseract failed: %s", exc)
        return ""


def _run_paddle(image_input):
    """Run PaddleOCR and flatten all detected line texts into a single string."""
    ocr = _get_paddle()
    if ocr is None:
        return ""
    try:
        if isinstance(image_input, str):
            result = ocr.ocr(image_input, cls=True)
        elif isinstance(image_input, np.ndarray):
            result = ocr.ocr(image_input, cls=True)
        elif isinstance(image_input, Image.Image):
            result = ocr.ocr(np.array(image_input), cls=True)
        else:
            result = ocr.ocr(np.array(image_input), cls=True)
        
        if result and result[0]:
            lines = [line[1][0] for line in result[0] if line and len(line) > 1]
            return " ".join(lines).strip()
        return ""
    except Exception as exc:
        logger.warning("PaddleOCR failed: %s", exc)
        return ""


def _arbitrate(tess_text, paddle_text):
    """
    Heuristic comparator for dual-engine OCR.
    
    Rules:
      1. If one engine returned nothing, use the other.
      2. similarity >= 80 -> use Tesseract (more structured)
      3. PaddleOCR output > 20% longer -> prefer Paddle.
      4. Otherwise -> use the longer output.
    """
    similarity = 0.0
    if fuzz is not None and tess_text and paddle_text:
        similarity = fuzz.ratio(tess_text, paddle_text)
    
    if not tess_text and not paddle_text:
        return _make_result("", "none", tess_text, paddle_text, similarity)
    
    if not tess_text:
        return _make_result(paddle_text, "paddle", tess_text, paddle_text, similarity)
    
    if not paddle_text:
        return _make_result(tess_text, "tesseract", tess_text, paddle_text, similarity)
    
    if similarity >= 80:
        return _make_result(tess_text, "tesseract", tess_text, paddle_text, similarity)
    
    # PaddleOCR significantly longer -> likely captured more content
    if len(paddle_text) > len(tess_text) * 1.2:
        return _make_result(paddle_text, "paddle", tess_text, paddle_text, similarity)
    
    # Fallback: use the longer output
    if len(paddle_text) > len(tess_text):
        return _make_result(paddle_text, "paddle", tess_text, paddle_text, similarity)
    
    return _make_result(tess_text, "tesseract", tess_text, paddle_text, similarity)


def _make_result(text, engine, tess_raw, paddle_raw, similarity):
    return {
        "text": text,
        "engine_used": engine,
        "tesseract_raw": tess_raw,
        "paddle_raw": paddle_raw,
        "similarity": similarity,
    }


def extract_text_safe(image_input):
    """
    Safe text extraction with automatic fallback.
    Tries dual-engine first, falls back to Tesseract only.
    """
    try:
        return extract_text(image_input)
    except Exception as e:
        # Fallback to Tesseract with full preprocessing
        logger.warning("Dual OCR failed, falling back to Tesseract: %s", e)
        preprocessed = preprocess_for_ocr(image_input)
        return pytesseract.image_to_string(preprocessed, config="--oem 3 --psm 6").strip()
