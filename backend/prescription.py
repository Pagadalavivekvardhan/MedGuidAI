import streamlit as st
from PIL import Image
import base64
import io
import json
import os
import traceback

# Import direct Groq client
try:
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)
    else:
        client = None
except ImportError:
    client = None

# Try to import API client for backend mode
try:
    from frontend.api_client import upload_prescription as api_upload_prescription
    from frontend.api_client import get_api_key, check_backend_health
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False

from backend.utils.image_preprocessing import enhance_for_vision_model
from backend.utils.rxnorm import correct_medicines_list


def _extract_via_api(image: Image.Image) -> list:
    """Extract medicines using the FastAPI backend with API key authentication."""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    result = api_upload_prescription(img_buffer.getvalue(), "prescription.png")
    return result.get("medicines", [])


def _extract_via_groq(image: Image.Image) -> list:
    """Extract medicines using direct Groq API call (fallback).

    Uses Groq chat.completions.create with llama-4-scout vision model.
    """
    if client is None:
        st.error("Groq client not available. Please configure GROQ_API_KEY.")
        return []

    processed_image = enhance_for_vision_model(image)

    img_buffer = io.BytesIO()
    processed_image.save(img_buffer, format="PNG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    prompt = """You are a PRESCRIPTION TRANSCRIPTION ASSISTANT.

Your ONLY job is to READ every piece of text visible on this prescription image and TRANSCRIBE it EXACTLY as written.

ABSOLUTE RULES - VIOLATION IS UNACCEPTABLE:

1. NEVER GUESS. NEVER INVENT. NEVER SUBSTITUTE. NEVER CORRECT.
   - If you see "Esmayo", write "Esmayo" — NOT "Asmacard" or any known drug.
   - If you see "FlupijaM", write "FlupijaM" — NOT "Fluconazole" or any known drug.
   - If you see "Elma", write "Elma" — NOT "Elmox" or any known drug.

2. TRANSCRIBE EVERY SINGLE WORD you can see on the prescription.
   - Every line, every scribble, every number.
   - If something looks like a duration (e.g., "5 days", "x7 days", "for 1 week", "10 days"), WRITE IT.
   - If something looks like frequency (e.g., "1-0-1", "BD", "TID", "OD", "SOS", "1x daily"), WRITE IT.
   - DO NOT skip any text just because you think it's unclear.

3. If you CANNOT read something at all, use "[ILLEGIBLE]" — do NOT guess.
   If you can read SOME letters, write those letters followed by "...": e.g., "Flu..."

4. For each medicine, capture EVERYTHING written near it:
   - Medicine name (exact spelling)
   - All numbers/dosages visible (e.g., 500mg, 10ml, 1 tab)
   - All frequency notations visible (1-0-1, BD, TID, OD, SOS, etc.)
   - All duration notations visible (x5 days, 7 days, continue, etc.)
   - All instructions visible (AC, PC, before food, after food, etc.)
   - All purpose/diagnosis text visible

5. Even if text is in a non-standard format, transcribe it as-is.
   - Doctor wrote "x10" → write "x10" for duration
   - Doctor wrote "1M-1N" → write "1M-1N" for frequency
   - Doctor wrote "tab x7" → write "tab" for dosage and "x7" for duration

For EACH medicine, return a JSON object:
- "name": EXACTLY what is written for the medicine name
- "dosage": ALL dosage/quantity text visible (e.g., "500mg", "1 tab BD", "10ml")
- "frequency": ALL frequency text visible (e.g., "1-0-1", "BD", "TID", "OD", "SOS", "as directed")
- "duration": ALL duration text visible (e.g., "5 days", "x7", "continue", "10 days", "1 month")
- "use": ALL purpose/diagnosis text visible (e.g., "for cough", "for infection", "fever")
- "instructions": ALL instruction text visible (e.g., "after food", "before bed", "on empty stomach")
- "raw_text": EVERYTHING else visible near this medicine that doesn't fit above
- "confidence": "high" / "medium" / "low"

Even if you are unsure about the CATEGORY, include the text in the most appropriate field.
The goal is ZERO information loss — capture everything.

Return ONLY a valid JSON array.

Example:
[
  {
    "name": "Paracetamol",
    "dosage": "500mg",
    "frequency": "1-0-1",
    "duration": "x5 days",
    "use": "for fever",
    "instructions": "after food",
    "raw_text": "",
    "confidence": "high"
  }
]

Return ONLY the JSON array."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.1,
        )

        raw_text = response.choices[0].message.content.strip()

        # Clean up markdown formatting
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        # Extract JSON array
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        if start != -1 and end > start:
            raw_text = raw_text[start:end]

        return json.loads(raw_text.strip())

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response: {e}")
        return []
    except Exception as e:
        st.error(f"Error analyzing prescription: {e}")
        traceback.print_exc()
        return []


def extract_medicines_from_image(image: Image.Image) -> list:
    """Extract medicine info from prescription image.

    Uses API backend with API key if configured, otherwise falls back to direct Groq calls.
    """
    use_api = (
        HAS_API_CLIENT
        and get_api_key()
        and check_backend_health()
    )

    try:
        if use_api:
            return _extract_via_api(image)
        else:
            return _extract_via_groq(image)
    except Exception as e:
        st.error(f"Error analyzing prescription: {e}")
        if use_api:
            st.info("Falling back to direct mode...")
            try:
                return _extract_via_groq(image)
            except Exception as e2:
                st.error(f"Direct mode also failed: {e2}")
        return []


def display_medicines(medicines: list):
    """Display extracted medicines in a nice format."""
    if not medicines:
        st.warning("No medicines could be extracted from the image.")
        return

    for i, med in enumerate(medicines, 1):
        with st.expander(f"💊 {med.get('name', f'Medicine {i}')}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Dosage:** {med.get('dosage', 'N/A')}")
                st.markdown(f"**Frequency:** {med.get('frequency', 'N/A')}")
                st.markdown(f"**Duration:** {med.get('duration', 'N/A')}")

            with col2:
                st.markdown(f"**For:** {med.get('use', 'N/A')}")
                if med.get("instructions"):
                    st.info(f"📋 {med['instructions']}")
                confidence = med.get('confidence', 'unknown')
                conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.caption(f"{conf_colors.get(confidence, '⚪')} Transcription confidence: {confidence}")


def prescription_tab():
    st.header("💊 AI Prescription Reader")
    st.markdown("Upload a prescription image and get detailed medicine information extracted using AI vision.")

    uploaded_file = st.file_uploader(
        "Upload Prescription Image", type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Prescription", use_column_width=True)

        if st.button("🔍 Extract Medicine Info", type="primary"):
            with st.spinner("Analyzing prescription with AI..."):
                medicines = extract_medicines_from_image(image)
                # Post-process: correct medicine names using RxNorm + fuzzy matching
                if medicines:
                    medicines = correct_medicines_list(medicines)
                st.session_state["prescription_result"] = medicines

    if "prescription_result" in st.session_state:
        medicines = st.session_state["prescription_result"]

        if medicines:
            st.success(f"✅ Successfully extracted {len(medicines)} medicine(s)")
            display_medicines(medicines)

            st.download_button(
                label="📥 Download as JSON",
                data=json.dumps(medicines, indent=2),
                file_name="prescription_data.json",
                mime="application/json",
            )
        else:
            st.warning("No medicines found. Please ensure the image is clear and contains prescription text.")
