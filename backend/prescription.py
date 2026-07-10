import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

def prescription_tab():

    st.header("💊 AI Prescription Reader")

    uploaded_file = st.file_uploader("Upload Prescription Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        if st.button("💊 Extract Medicine Info"):

            with st.spinner("Analyzing prescription..."):

                model = genai.GenerativeModel("gemini-2.5-flash")

                prompt = """
You are a medical assistant.

Analyze this prescription image.

Extract medicines and explain them.

Return in this EXACT format for each medicine (use bullet points):

**Medicine Name**
* **Dosage:** ...
* **Frequency:** ...
* **Duration:** ...
* **Use:** short simple explanation (1 line)

Leave a blank line between medicines.

Rules:
- Convert 1-0-1 → twice daily
- sos → as needed
- Use simple English
- Keep explanation short
"""

                response = model.generate_content([prompt, image])
                st.session_state["prescription_result"] = response.text

        if "prescription_result" in st.session_state:
            result = st.session_state["prescription_result"]
            
            # ✅ CLEAN DISPLAY (NO JSON)
            st.subheader("💊 Medicines")

            medicines = result.split("\n\n")  # split by blank line

            for med in medicines:
                if med.strip():
                    st.markdown(f"### 💊 {med.strip()}")