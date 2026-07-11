"""
rxnorm.py
---------
Medical Integrity and Safety Layer

Validates every drug name extracted by OCR against the NIH RxNorm database.
Uses RapidFuzz for autonomous correction of OCR-induced misspellings before
forwarding validated drug data.

Pipeline per drug name:
  1. Direct RxNorm API lookup
  2. If no match: fuzzy-correct against a curated common drug list, then retry
  3. Return validated canonical name + RxCUI, or flag as unvalidated
"""

import logging
import os
import re
from typing import Optional, List, Dict

import requests
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# ─── Configuration (inline, no config.py dependency) ─────────────────────────
RXNORM_BASE_URL = os.getenv("RXNORM_BASE_URL", "https://rxnav.nlm.nih.gov/REST")
RXNORM_TIMEOUT = int(os.getenv("RXNORM_TIMEOUT", "5"))
FUZZY_CORRECTION_THRESHOLD = int(os.getenv("FUZZY_CORRECTION_THRESHOLD", "60"))


# ─── Common drug reference list for fuzzy correction ─────────────────────────
COMMON_DRUGS: List[str] = [
    # Pain & Fever
    "Paracetamol", "Acetaminophen", "Aspirin", "Ibuprofen", "Diclofenac",
    "Naproxen", "Tramadol", "Codeine", "Morphine", "Pentazocine",
    "Celecoxib", "Etoricoxib", "Aceclofenac", "Piroxicam",
    # Antibiotics
    "Amoxicillin", "Amoxicillin-Clavulanate", "Azithromycin", "Ciprofloxacin",
    "Levofloxacin", "Doxycycline", "Cefuroxime", "Cefixime", "Ceftriaxone",
    "Clindamycin", "Metronidazole", "Norfloxacin", "Ofloxacin",
    "Co-trimoxazole", "Fluconazole", "Acyclovir",
    # Gastrointestinal
    "Omeprazole", "Pantoprazole", "Esomeprazole", "Ranitidine", "Pantoprazole",
    "Domperidone", "Ondansetron", "Metoclopramide", "Sucralfate", "Lansoprazole",
    "Rabeprazole", "Itopride", "Palitaprazole",
    # Cardiovascular
    "Atorvastatin", "Rosuvastatin", "Amlodipine", "Losartan", "Telmisartan",
    "Valsartan", "Ramipril", "Lisinopril", "Enalapril", "Metoprolol",
    "Bisoprolol", "Carvedilol", "Digoxin", "Warfarin", "Clopidogrel",
    "Rivaroxaban", "Apixaban", "Hydrochlorothiazide", "Furosemide",
    "Spironolactone", "Chlorthalidone", "Sildenafil", "Nitroglycerin",
    # Diabetes
    "Metformin", "Glimepiride", "Gliclazide", "Glipizide", "Sitagliptin",
    "Empagliflozin", "Dapagliflozin", "Insulin", "Pioglitazone",
    "Voglibose", "Acarbose",
    # Respiratory
    "Salbutamol", "Montelukast", "Budesonide", "Fluticasone", "Levosalbutamol",
    "Theophylline", "Tiotropium", "Formoterol", "Beclomethasone",
    # Allergy / Anti-histamine
    "Cetirizine", "Levocetirizine", "Loratadine", "Fexofenadine",
    "Chlorpheniramine", "Diphenhydramine", "Promethazine",
    # Psychiatry / Neurology
    "Amitriptyline", "Sertraline", "Escitalopram", "Fluoxetine", "Duloxetine",
    "Venlafaxine", "Alprazolam", "Clonazepam", "Zolpidem", "Diazepam",
    "Lorazepam", "Quetiapine", "Olanzapine", "Risperidone", "Haloperidol",
    "Phenytoin", "Valproate", "Levetiracetam", "Carbamazepine", "Gabapentin",
    "Pregabalin", "Lamotrigine",
    # Thyroid
    "Levothyroxine", "Carbimazole", "Methimazole",
    # Vitamins / Supplements
    "Ferrous Sulfate", "Folic Acid", "Calcium Carbonate", "Vitamin D3",
    "Cholecalciferol", "Methylcobalamin", "Cyanocobalamin", "Zinc",
    "Iron", "Multivitamin",
    # Steroids
    "Prednisolone", "Prednisone", "Dexamethasone", "Methylprednisolone",
    "Betamethasone",
    # Other
    "Oseltamivir", "Hydroxychloroquine", "Ivermectin", "Azathioprine",
    "Mycophenolate", "Cyclosporine", "Tacrolimus",
]


# ─── Public API ──────────────────────────────────────────────────────────────

def validate_drug_name(raw_drug_name: str) -> Dict:
    """Validate a single drug name against RxNorm with fuzzy correction.

    Two-pass validation:
      Pass 1: Direct RxNorm lookup
      Pass 2: Fuzzy-correct misspelling, then retry RxNorm

    Args:
        raw_drug_name: Drug name string (potentially misspelled by OCR).

    Returns:
        Dict with keys: validated, canonical_name, rxcui, corrected_from, raw
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
            logger.info("Corrected '%s' -> '%s' (RxCUI: %s)", raw_drug_name, correction, result["rxcui"])
            return result

    logger.debug("Drug not validated: '%s'", raw_drug_name)
    return {"validated": False, "raw": raw_drug_name}


def correct_medicine_name(raw_name: str) -> str:
    """Simple convenience function: correct a medicine name using fuzzy matching.

    Returns the corrected name if a good match is found, otherwise returns the original.

    Args:
        raw_name: Potentially misspelled medicine name.

    Returns:
        Corrected or original medicine name.
    """
    if not raw_name or not raw_name.strip():
        return raw_name

    # Try direct RxNorm lookup first
    result = _validate_rxnorm(raw_name)
    if result and result.get("canonical_name"):
        return result["canonical_name"]

    # Try fuzzy correction
    correction = _fuzzy_correct(raw_name)
    if correction:
        return correction

    return raw_name


def correct_medicines_list(medicines: List[Dict]) -> List[Dict]:
    """Post-process a list of medicine dicts, correcting names using RxNorm + fuzzy matching.

    For each medicine:
      - Corrects the 'name' field
      - Adds 'corrected_from' if name was changed
      - Adds 'rxnorm_validated': True/False

    Args:
        medicines: List of medicine dicts with at least a 'name' key.

    Returns:
        Corrected list of medicine dicts.
    """
    corrected = []
    for med in medicines:
        original_name = med.get("name", "")
        if not original_name or original_name.startswith("["):
            # Don't correct [UNCERTAIN] or [ILLEGIBLE] markers
            med["rxnorm_validated"] = False
            corrected.append(med)
            continue

        result = validate_drug_name(original_name)

        if result.get("validated") and result.get("canonical_name"):
            med["corrected_from"] = original_name
            med["name"] = result["canonical_name"]
            if result.get("rxcui"):
                med["rxcui"] = result["rxcui"]
            med["rxnorm_validated"] = True
        else:
            # Try fuzzy correction without RxNorm
            fuzzy = _fuzzy_correct(original_name)
            if fuzzy:
                med["corrected_from"] = original_name
                med["name"] = fuzzy
                med["rxnorm_validated"] = False
            else:
                med["rxnorm_validated"] = False

        corrected.append(med)

    return corrected


# ─── RxNorm API ──────────────────────────────────────────────────────────────

def _validate_rxnorm(drug_name: str) -> Optional[Dict]:
    """Lookup a drug name in the NIH RxNorm API."""
    try:
        url = f"{RXNORM_BASE_URL}/rxcui.json"
        params = {"name": drug_name, "search": 1}
        response = requests.get(url, params=params, timeout=RXNORM_TIMEOUT)
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
    """Fetch the canonical drug name for a given RxCUI."""
    try:
        url = f"{RXNORM_BASE_URL}/rxcui/{rxcui}/property.json"
        params = {"propName": "RxNorm Name"}
        response = requests.get(url, params=params, timeout=RXNORM_TIMEOUT)
        data = response.json()
        props = data.get("propConceptGroup", {}).get("propConcept", [])
        if props:
            return props[0].get("propValue")
    except Exception:
        pass
    return None


# ─── Fuzzy correction ────────────────────────────────────────────────────────

def _fuzzy_correct(raw_name: str) -> Optional[str]:
    """Find the closest match in COMMON_DRUGS using RapidFuzz WRatio scorer."""
    if not raw_name or len(raw_name) < 3:
        return None

    match = process.extractOne(
        raw_name,
        COMMON_DRUGS,
        scorer=fuzz.WRatio,
        score_cutoff=FUZZY_CORRECTION_THRESHOLD,
    )
    if match:
        best_name, score, _ = match
        logger.debug("Fuzzy: '%s' -> '%s' (score %.1f)", raw_name, best_name, score)
        return best_name
    return None


def validate_test_name(raw_name: str) -> str:
    """Correct common OCR errors in lab test names using fuzzy matching.

    This is a lightweight version for lab report test names (not drug names).

    Args:
        raw_name: Test name string (potentially OCR-mangled).

    Returns:
        Corrected test name, or original if no match found.
    """
    COMMON_TESTS = [
        "Hemoglobin", "Hematocrit", "WBC", "White Blood Cell", "RBC",
        "Red Blood Cell", "Platelet", "Platelet Count", "MPV",
        "Mean Corpuscular Volume", "MCV", "MCH", "MCHC",
        "Fasting Glucose", "Blood Glucose", "HbA1c", "HbA1C",
        "Total Cholesterol", "HDL", "LDL", "Triglycerides",
        "Creatinine", "Blood Urea Nitrogen", "BUN", "Uric Acid",
        "SGOT", "SGPT", "ALT", "AST", "ALP", "Bilirubin",
        "Total Protein", "Albumin", "Globulin",
        "Sodium", "Potassium", "Chloride", "Calcium", "Magnesium",
        "Phosphorus", "Iron", "Ferritin", "TIBC",
        "TSH", "Free T3", "Free T4", "Thyroid Stimulating Hormone",
        "Vitamin D", "Vitamin B12", "Folate",
        "CRP", "ESR", "Rheumatoid Factor",
        "PSA", "AFP", "CEA", "CA-125",
    ]

    if not raw_name or len(raw_name) < 4:
        return raw_name

    match = process.extractOne(
        raw_name,
        COMMON_TESTS,
        scorer=fuzz.WRatio,
        score_cutoff=70,
    )
    if match:
        best_name, score, _ = match
        logger.debug("Test name fuzzy: '%s' -> '%s' (score %.1f)", raw_name, best_name, score)
        return best_name
    return raw_name
