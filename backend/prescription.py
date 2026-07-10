import streamlit as st
from groq import Groq
from PIL import Image
import base64
import io
import os

# Groq API Configuration
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=api_key)

def prescription_tab():

    st.header("💊 AI Prescription Reader")

    uploaded_file = st.file_uploader("Upload Prescription Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        if st.button("💊 Extract Medicine Info"):

            with st.spinner("Analyzing prescription..."):

                # Convert image to base64 for Groq API
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

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

                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    max_tokens=4096
                )
                st.session_state["prescription_result"] = response.choices[0].message.content

        if "prescription_result" in st.session_state:
            result = st.session_state["prescription_result"]
            
            # ✅ CLEAN DISPLAY (NO JSON)
            st.subheader("💊 Medicines")

            medicines = result.split("\n\n")  # split by blank line

            for med in medicines:
                if med.strip():
                    st.markdown(f"### 💊 {med.strip()}")