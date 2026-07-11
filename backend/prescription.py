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

    prompt = """You are an expert medical assistant specializing in prescription analysis.

Analyze this prescription image carefully and extract ALL medicine information.

For EACH medicine found, return a JSON object with these fields:
- "name": The medicine name (exact spelling as written)
- "dosage": The dosage (e.g., "500mg", "10mg", "5ml")
- "frequency": How often to take (e.g., "Twice daily", "Once daily", "Three times daily", "As needed")
- "duration": How long to take (e.g., "5 days", "1 week", "2 weeks", "Continue")
- "use": What it's for (simple explanation)
- "instructions": Any special instructions (e.g., "Take after food", "Take before bedtime")

IMPORTANT RULES:
1. Convert doctor shorthand: "1-0-1" = Twice daily, "1-1-1" = Three times daily, "sos" = As needed
2. Extract EVERY medicine visible in the image
3. Be precise with dosage numbers
4. Preserve the exact spelling as written on the prescription
5. Return ONLY a valid JSON array, no other text

Example output format:
[
  {
    "name": "Paracetamol",
    "dosage": "500mg",
    "frequency": "Three times daily",
    "duration": "5 days",
    "use": "Pain relief and fever reduction",
    "instructions": "Take after food"
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
