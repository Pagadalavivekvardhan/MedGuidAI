"""
ocr_service.py
--------------
Stage 2: Dual-Engine OCR Voting System

Runs PaddleOCR and Tesseract in parallel on a preprocessed image.
A heuristic arbitration layer compares both outputs:
  - If similarity >= threshold  → accept Tesseract output (more structured)
  - If PaddleOCR is significantly longer → prefer PaddleOCR
  - Otherwise                   → invoke GPT-4o-mini as tiebreaker

Returns the best-quality text string for downstream processing.
"""

import logging
from typing import Optional

import pytesseract
from rapidfuzz import fuzz

# Support environments where the project root isn't on sys.path (e.g. some linters/IDEs)
try:
    from services.opencv_service import preprocess_image, preprocess_to_pil
except ImportError:
    import os
    import sys

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from services.opencv_service import preprocess_image, preprocess_to_pil
from config import Config

logger = logging.getLogger(__name__)

# Lazy-load PaddleOCR to avoid slow import at startup
_paddle_ocr = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddle_ocr


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(image_path: str) -> dict:
    """
    Full dual-engine OCR pipeline for a document image.

    Args:
        image_path: Path to the raw (unprocessed) image file.

    Returns:
        dict with keys:
          - text         : final chosen OCR text
          - engine_used  : "tesseract" | "paddle" | "llm_arbitration"
          - tesseract_raw: raw Tesseract output
          - paddle_raw   : raw PaddleOCR output
          - similarity   : fuzz.ratio score between both outputs
    """
    # Step 1 – Preprocess
    pil_image = preprocess_to_pil(image_path)
    preprocessed_arr = preprocess_image(image_path)

    # Step 2 – Tesseract
    tess_text = _run_tesseract(pil_image)

    # Step 3 – PaddleOCR
    paddle_text = _run_paddle(preprocessed_arr)

    # Step 4 – Arbitration
    result = _arbitrate(tess_text, paddle_text)
    logger.info(
        "OCR complete. Engine: %s | Similarity: %.1f",
        result["engine_used"],
        result["similarity"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Individual engines
# ─────────────────────────────────────────────────────────────────────────────

def _run_tesseract(pil_image) -> str:
    """Run Tesseract OCR with page segmentation mode 6 (uniform block of text)."""
    try:
        text = pytesseract.image_to_string(pil_image, config="--psm 6")
        return text.strip()
    except Exception as exc:
        logger.warning("Tesseract failed: %s", exc)
        return ""


def _run_paddle(preprocessed_arr) -> str:
    """Run PaddleOCR and flatten all detected line texts into a single string."""
    try:
        ocr = _get_paddle()
        result = ocr.ocr(preprocessed_arr, cls=True)
        if result and result[0]:
            lines = [line[1][0] for line in result[0] if line and len(line) > 1]
            return " ".join(lines).strip()
        return ""
    except Exception as exc:
        logger.warning("PaddleOCR failed: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Arbitration
# ─────────────────────────────────────────────────────────────────────────────

def _arbitrate(tess_text: str, paddle_text: str) -> dict:
    """
    Heuristic comparator + LLM tiebreaker.

    Rules:
      1. If one engine returned nothing, use the other.
      2. similarity >= OCR_SIMILARITY_THRESHOLD  → use Tesseract.
      3. PaddleOCR output > 20% longer than Tesseract → prefer Paddle.
      4. Otherwise call GPT as tiebreaker.
    """
    similarity = fuzz.ratio(tess_text, paddle_text)

    if not tess_text and not paddle_text:
        return _make_result("", "none", tess_text, paddle_text, similarity)

    if not tess_text:
        return _make_result(paddle_text, "paddle", tess_text, paddle_text, similarity)

    if not paddle_text:
        return _make_result(tess_text, "tesseract", tess_text, paddle_text, similarity)

    if similarity >= Config.OCR_SIMILARITY_THRESHOLD:
        return _make_result(tess_text, "tesseract", tess_text, paddle_text, similarity)

    # PaddleOCR significantly longer → likely captured more content
    if len(paddle_text) > len(tess_text) * 1.2:
        return _make_result(paddle_text, "paddle", tess_text, paddle_text, similarity)

    # Fallback: LLM arbitration
    best = _llm_arbitrate(tess_text, paddle_text)
    return _make_result(best, "llm_arbitration", tess_text, paddle_text, similarity)


def _llm_arbitrate(tess_text: str, paddle_text: str) -> str:
    """
    Uses GPT-4o-mini to choose the more medically accurate OCR output
    when both engines produce significantly different results.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        prompt = f"""You are an OCR arbitration assistant for medical documents.
Two OCR engines produced different outputs from the same medical document image.
Select the output that is more likely to be accurate for a medical prescription or lab report.
Return ONLY the selected text with no explanation.

Engine A (Tesseract):
{tess_text}

Engine B (PaddleOCR):
{paddle_text}"""

        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM arbitration failed, defaulting to Tesseract: %s", exc)
        return tess_text


def _make_result(text, engine, tess_raw, paddle_raw, similarity) -> dict:
    return {
        "text": text,
        "engine_used": engine,
        "tesseract_raw": tess_raw,
        "paddle_raw": paddle_raw,
        "similarity": similarity,
    }
