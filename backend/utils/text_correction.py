"""
text_correction.py
------------------
RapidFuzz Text Correction for Medical Documents

Corrects OCR-induced misspellings in medical text using fuzzy matching.
Common medical terms, drug names, and lab values are corrected
before being sent to the LLM for analysis.
"""

import logging
import re
from typing import Optional, List, Dict

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

logger = logging.getLogger(__name__)

# Common medical terms and abbreviations
MEDICAL_TERMS = {
    # Lab tests
    "hemoglobin": "hemoglobin", "haemoglobin": "hemoglobin",
    "glucose": "glucose", "blod sugar": "blood sugar",
    "cholesterol": "cholesterol", "triglycerides": "triglycerides",
    "creatinine": "creatinine", "urea": "urea",
    "bilirubin": "bilirubin", "albumin": "albumin",
    "sodium": "sodium", "potassium": "potassium",
    "calcium": "calcium", "magnesium": "magnesium",
    "phosphorus": "phosphorus", "chloride": "chloride",
    "platelet": "platelet", "platelets": "platelets",
    "wbc": "white blood cells", "rbc": "red blood cells",
    "hct": "hematocrit", "mcv": "mean corpuscular volume",
    "mch": "mean corpuscular hemoglobin",
    "mchc": "mean corpuscular hemoglobin concentration",
    "rdw": "red cell distribution width",
    "esr": "erythrocyte sedimentation rate",
    "crp": "c-reactive protein",
    "tsh": "thyroid stimulating hormone",
    "ft3": "free t3", "ft4": "free t4",
    "hba1c": "glycated hemoglobin", "hbA1c": "glycated hemoglobin",
    "alt": "alanine transaminase", "ast": "aspartate transaminase",
    "alp": "alkaline phosphatase",
    "ggt": "gamma glutamyl transferase",
    "lft": "liver function test", "kft": "kidney function test",
    "cbc": "complete blood count", "cbp": "complete blood picture",
    # Common misspellings from OCR
    "hemglobin": "hemoglobin", "hemogobin": "hemoglobin",
    "hemogloin": "hemoglobin", "haemogloin": "hemoglobin",
    "platlet": "platelet", "platlets": "platelets",
    "leukocyte": "white blood cells",
    "erythrocyte": "red blood cells",
}

# Common drug names for prescription correction
DRUG_NAMES = [
    "Paracetamol", "Ibuprofen", "Amoxicillin", "Azithromycin",
    "Metformin", "Atorvastatin", "Amlodipine", "Losartan",
    "Lisinopril", "Metoprolol", "Omeprazole", "Pantoprazole",
    "Ranitidine", "Cetirizine", "Montelukast", "Salbutamol",
    "Prednisolone", "Dexamethasone", "Ciprofloxacin",
    "Levofloxacin", "Cefuroxime", "Doxycycline",
    "Clindamycin", "Metronidazole", "Fluconazole", "Acyclovir",
    "Insulin", "Glimepiride", "Sitagliptin", "Empagliflozin",
    "Rosuvastatin", "Telmisartan", "Valsartan", "Ramipril",
    "Bisoprolol", "Carvedilol", "Digoxin", "Warfarin",
    "Clopidogrel", "Rivaroxaban", "Apixaban",
    "Furosemide", "Spironolactone", "Hydrochlorothiazide",
    "Levothyroxine", "Carbimazole", "Ferrous Sulfate",
    "Folic Acid", "Calcium Carbonate", "Vitamin D3",
    "Cholecalciferol", "Methylcobalamin", "Gabapentin",
    "Pregabalin", "Amitriptyline", "Sertraline",
    "Escitalopram", "Fluoxetine", "Alprazolam",
    "Clonazepam", "Zolpidem", "Quetiapine",
    "Haloperidol", "Risperidone", "Phenytoin",
    "Levetiracetam", "Tramadol", "Codeine", "Morphine",
    "Ondansetron", "Domperidone", "Metoclopramide",
    "Esomeprazole", "Sucralfate",
]


def correct_ocr_text(text: str) -> str:
    """
    Correct common OCR errors in medical text.
    
    Args:
        text: Raw OCR text
    
    Returns:
        Corrected text
    """
    if not text or not text.strip():
        return text
    
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    
    # Correct common OCR errors
    corrected = _correct_common_errors(text)
    
    # Correct medical terms
    corrected = _correct_medical_terms(corrected)
    
    return corrected.strip()


def correct_medical_terms(text: str) -> str:
    """
    Correct medical terms using fuzzy matching.
    
    Args:
        text: OCR text
    
    Returns:
        Text with corrected medical terms
    """
    if not text or process is None:
        return text
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        clean_word = re.sub(r"[^a-zA-Z0-9]", "", word.lower())
        if len(clean_word) < 3:
            corrected_words.append(word)
            continue
        
        # Check if it matches a medical term
        match = process.extractOne(
            clean_word,
            MEDICAL_TERMS.keys(),
            scorer=fuzz.ratio,
            score_cutoff=80,
        )
        
        if match:
            best_key, score, _ = match
            corrected = MEDICAL_TERMS[best_key]
            # Preserve original case style
            if word[0].isupper():
                corrected = corrected.capitalize()
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)


def correct_drug_name(name: str) -> Optional[str]:
    """
    Correct a drug name using fuzzy matching.
    
    Args:
        name: Potentially misspelled drug name
    
    Returns:
        Corrected drug name or None if no good match
    """
    if not name or process is None:
        return None
    
    match = process.extractOne(
        name,
        DRUG_NAMES,
        scorer=fuzz.WRatio,
        score_cutoff=75,
    )
    
    if match:
        best_name, score, _ = match
        logger.debug("Fuzzy: '%s' -> '%s' (score %.1f)", name, best_name, score)
        return best_name
    return None


def _correct_common_errors(text: str) -> str:
    """
    Fix common OCR errors that affect medical documents.
    """
    # Common character substitutions
    # Fix common misspellings
    common_fixes = {
        "blod": "blood",
        "hemoglobin": "hemoglobin",
        "platlet": "platelet",
        "platlets": "platelets",
        "leukocyte": "white blood cells",
        "erythrocyte": "red blood cells",
        "thrombocite": "platelet",
        "thrombocytes": "platelets",
    }
    
    for wrong, correct in common_fixes.items():
        text = re.sub(rf"\b{wrong}\b", correct, text, flags=re.IGNORECASE)
    
    return text
