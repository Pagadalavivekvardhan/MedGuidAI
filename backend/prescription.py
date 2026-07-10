import streamlit as st
from groq import Groq
from PIL import Image
import base64
import io
import json
from dotenv import load_dotenv
import os

from backend.utils.image_preprocessing import enhance_for_vision_model

# Load environment variables
load_dotenv()

# Groq API Configuration
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=api_key)

def extract_medicines_from_image(image: Image.Image) -> list:
    """Use Groq vision model to extract medicine info from prescription image."""
    processed_image = enhance_for_vision_model(image)

    img_buffer = io.BytesIO()
    processed_image.save(img_buffer, format="PNG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    prompt = """You are an expert medical assistant specializing in prescription analysis.

Analyze this prescription image carefully and extract ALL medicine information.

For EACH medicine found, return a JSON object with these fields:
- "name": The medicine name (exact spelling)
- "dosage": The dosage (e.g., "500mg", "10mg", "5ml")
- "frequency": How often to take (e.g., "Twice daily", "Once daily", "Three times daily", "As needed")
- "duration": How long to take (e.g., "5 days", "1 week", "2 weeks", "Continue")
- "use": What it's for (simple explanation)
- "instructions": Any special instructions (e.g., "Take after food", "Take before bedtime")

IMPORTANT RULES:
1. Convert doctor shorthand: "1-0-1" = Twice daily, "1-1-1" = Three times daily, "sos" = As needed
2. Extract EVERY medicine visible in the image
3. Be precise with dosage numbers
4. Return ONLY a valid JSON array, no other text

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

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

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
        st.image(image, caption="Uploaded Prescription", use_container_width=True)

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