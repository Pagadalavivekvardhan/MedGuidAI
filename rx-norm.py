"""
rxnorm_service.py
-----------------
Stage 3: Medical Integrity and Safety Layer

Validates every drug name extracted by OCR against the NIH RxNorm database.
Uses RapidFuzz for autonomous correction of OCR-induced misspellings before
forwarding validated drug data to the LLM.

Pipeline per drug name:
  1. Direct RxNorm API lookup
  2. If no match: fuzzy-correct against a curated common drug list, then retry
  3. Return validated canonical name + RxCUI, or flag as unvalidated
"""

import logging
import re
from typing import Optional

import requests
from rapidfuzz import process, fuzz

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Common drug reference list for fuzzy correction
# This list covers frequently prescribed drugs; extend as needed.
# ─────────────────────────────────────────────────────────────────────────────
COMMON_DRUGS = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Amoxicillin", "Azithromycin",
    "Metformin", "Atorvastatin", "Amlodipine", "Losartan", "Lisinopril",
    "Metoprolol", "Omeprazole", "Pantoprazole", "Ranitidine", "Cetirizine",
    "Montelukast", "Salbutamol", "Prednisolone", "Dexamethasone", "Ciprofloxacin",
    "Levofloxacin", "Cefuroxime", "Amoxicillin-Clavulanate", "Doxycycline",
    "Clindamycin", "Metronidazole", "Fluconazole", "Acyclovir", "Oseltamivir",
    "Insulin", "Glimepiride", "Sitagliptin", "Empagliflozin", "Rosuvastatin",
    "Telmisartan", "Valsartan", "Ramipril", "Bisoprolol", "Carvedilol",
    "Digoxin", "Warfarin", "Clopidogrel", "Rivaroxaban", "Apixaban",
    "Furosemide", "Spironolactone", "Hydrochlorothiazide", "Chlorthalidone",
    "Levothyroxine", "Carbimazole", "Ferrous Sulfate", "Folic Acid",
    "Calcium Carbonate", "Vitamin D3", "Cholecalciferol", "Methylcobalamin",
    "Gabapentin", "Pregabalin", "Amitriptyline", "Sertraline", "Escitalopram",
    "Fluoxetine", "Alprazolam", "Clonazepam", "Zolpidem", "Quetiapine",
    "Haloperidol", "Risperidone", "Phenytoin", "Valproate", "Levetiracetam",
    "Tramadol", "Codeine", "Morphine", "Ondansetron", "Domperidone",
    "Metoclopramide", "Ranitidine", "Esomeprazole", "Sucralfate",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_drugs_from_text(ocr_text: str) -> list[dict]:
    """
    Main entry point. Extracts candidate drug names from OCR text,
    validates each against RxNorm, and returns a list of validation results.

    Args:
        ocr_text: Raw OCR-extracted text string.

    Returns:
        List of dicts:
          {canonical_name, rxcui, validated, corrected_from (optional), raw}
    """
    candidates = _extract_drug_candidates(ocr_text)
    results = []
    for name in candidates:
        result = correct_and_validate(name)
        results.append(result)
    # Deduplicate by canonical_name / rxcui
    seen = set()
    unique = []
    for r in results:
        key = r.get("rxcui") or r.get("raw", "").lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def correct_and_validate(raw_drug_name: str) -> dict:
    """
    Two-pass validation:
      Pass 1 – Direct RxNorm lookup
      Pass 2 – Fuzzy-correct misspelling, then retry RxNorm

    Args:
        raw_drug_name: Drug name string (potentially misspelled by OCR).

    Returns:
        Validation result dict.
    """
    # Pass 1: direct lookup
    result = _validate_rxnorm(raw_drug_name)
    if result:
        return result

    # Pass 2: fuzzy correction
    correction = _fuzzy_correct(raw_drug_name)
    if correction:
        result = _validate_rxnorm(correction)
        if result:
            result["corrected_from"] = raw_drug_name
            logger.info("Corrected '%s' → '%s' (RxCUI: %s)", raw_drug_name, correction, result["rxcui"])
            return result

    logger.debug("Drug not validated: '%s'", raw_drug_name)
    return {"validated": False, "raw": raw_drug_name}


# ─────────────────────────────────────────────────────────────────────────────
# RxNorm API
# ─────────────────────────────────────────────────────────────────────────────

def _validate_rxnorm(drug_name: str) -> Optional[dict]:
    """
    Looks up a drug name in the NIH RxNorm API.
    Returns a validated result dict, or None if not found.
    """
    try:
        url = f"{Config.RXNORM_BASE_URL}/rxcui.json"
        params = {"name": drug_name, "search": 1}
        response = requests.get(url, params=params, timeout=Config.RXNORM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
        if rxcui_list:
            rxcui = rxcui_list[0]
            canonical = _get_canonical_name(rxcui) or drug_name
            return {
                "validated": True,
                "rxcui": rxcui,
                "canonical_name": canonical,
                "raw": drug_name,
            }
    except requests.exceptions.Timeout:
        logger.warning("RxNorm API timeout for '%s'", drug_name)
    except Exception as exc:
        logger.warning("RxNorm lookup failed for '%s': %s", drug_name, exc)
    return None


def _get_canonical_name(rxcui: str) -> Optional[str]:
    """Fetches the canonical drug name for a given RxCUI."""
    try:
        url = f"{Config.RXNORM_BASE_URL}/rxcui/{rxcui}/property.json"
        params = {"propName": "RxNorm Name"}
        response = requests.get(url, params=params, timeout=Config.RXNORM_TIMEOUT)
        data = response.json()
        props = data.get("propConceptGroup", {}).get("propConcept", [])
        if props:
            return props[0].get("propValue")
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy correction
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_correct(raw_name: str) -> Optional[str]:
    """
    Finds the closest match in COMMON_DRUGS using RapidFuzz WRatio scorer.
    Returns the best match if it exceeds the configured threshold.
    """
    match = process.extractOne(
        raw_name,
        COMMON_DRUGS,
        scorer=fuzz.WRatio,
        score_cutoff=Config.FUZZY_CORRECTION_THRESHOLD,
    )
    if match:
        best_name, score, _ = match
        logger.debug("Fuzzy: '%s' → '%s' (score %.1f)", raw_name, best_name, score)
        return best_name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Drug name extraction from OCR text
# ─────────────────────────────────────────────────────────────────────────────

def _extract_drug_candidates(text: str) -> list[str]:
    """
    Heuristic extraction of candidate drug names from raw OCR text.

    Strategy:
      - Split text into tokens/lines.
      - Filter out pure numbers, very short tokens, and common non-drug words.
      - Return a deduplicated list of capitalised candidate tokens.

    Note: This is a lightweight heuristic. For production, a dedicated
    Named Entity Recognition (NER) model (e.g., scispaCy med7) is recommended.
    """
    STOPWORDS = {
        "the", "and", "for", "tab", "tablet", "cap", "capsule", "syrup",
        "injection", "inj", "mg", "ml", "once", "twice", "daily", "times",
        "morning", "night", "with", "after", "before", "meals", "food",
        "weeks", "days", "months", "dose", "doses", "dr", "rx", "name",
        "patient", "date", "age", "sex", "male", "female", "signature",
    }

    candidates = []
    seen = set()

    for line in text.splitlines():
        # Split on whitespace, slashes, commas
        tokens = re.split(r"[\s,/]+", line)
        for token in tokens:
            # Remove non-alphanumeric suffix (e.g. trailing dots, colons)
            clean = re.sub(r"[^a-zA-Z\-]", "", token).strip()
            if len(clean) < 4:
                continue
            if clean.lower() in STOPWORDS:
                continue
            # Must start with a letter
            if not clean[0].isalpha():
                continue
            key = clean.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(clean.capitalize())

    return candidates
