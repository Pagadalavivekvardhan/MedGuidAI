# Backend utilities package
# Lazy imports for convenience - modules loaded on first use

def __getattr__(name):
    if name == "extract_text":
        from .ocr_engine import extract_text
        return extract_text
    elif name == "extract_text_dual":
        from .ocr_engine import extract_text_dual
        return extract_text_dual
    elif name == "preprocess_image":
        from .image_preprocessing import preprocess_image
        return preprocess_image
    elif name == "preprocess_for_ocr":
        from .image_preprocessing import preprocess_for_ocr
        return preprocess_for_ocr
    elif name == "correct_ocr_text":
        from .text_correction import correct_ocr_text
        return correct_ocr_text
    elif name == "correct_medical_terms":
        from .text_correction import correct_medical_terms
        return correct_medical_terms
    elif name == "extract_text_safe":
        from .ocr_engine import extract_text_safe
        return extract_text_safe
    elif name == "correct_drug_name":
        from .text_correction import correct_drug_name
        return correct_drug_name
    raise AttributeError(f"module backend.utils has no attribute {name}")
