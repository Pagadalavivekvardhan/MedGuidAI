
import os
import json
import re
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import pytesseract

import google.generativeai as genai


########################################################
# Gemini Configuration
########################################################

# Recommended:
# Store your API key in .streamlit/secrets.toml
#
# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Temporary (Environment Variable)
model = None

def initialize_gemini():
    global model

    if model is None:
        api_key = "google_api_key"  # Replace with your actual API key

        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("Gemini API Key not found.")
                return False

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

    return True


########################################################
# Tesseract Configuration
########################################################

# Windows Path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


########################################################
# Image Processing
########################################################

def preprocess_image(image):

    img = np.array(image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    gray = cv2.fastNlMeansDenoising(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    kernel = np.ones((1, 1), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    return thresh


########################################################
# OCR
########################################################

def extract_text(image):

    processed = preprocess_image(image)

    config = (
        "--oem 3 "
        "--psm 6"
    )

    text = pytesseract.image_to_string(
        processed,
        config=config
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


########################################################
# Utility Functions
########################################################

def clean_value(value):

    if value is None:
        return ""

    return str(value).strip()


def safe_float(value):

    try:
        value = str(value)

        value = value.replace(",", "")
        value = value.replace("<", "")
        value = value.replace(">", "")

        return float(value)

    except:

        return None


def status_color(status):

    status = status.lower()

    if status == "high":
        return "🔴 High"

    if status == "low":
        return "🟠 Low"

    return "🟢 Normal"


########################################################
# Display Helpers
########################################################

def show_patient(patient):

    st.subheader("Patient Details")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Name",
        patient.get("name", "N/A")
    )

    c2.metric(
        "Age",
        patient.get("age", "N/A")
    )

    c3.metric(
        "Gender",
        patient.get("gender", "N/A")
    )


def show_tests(tests):

    st.subheader("Laboratory Results")

    if len(tests) == 0:

        st.warning("No laboratory values detected.")

        return

    rows = []

    for test in tests:

        rows.append({

            "Test":
            clean_value(test.get("name")),

            "Value":
            clean_value(test.get("value")),

            "Unit":
            clean_value(test.get("unit")),

            "Reference":
            clean_value(test.get("reference_range")),

            "Status":
            status_color(
                clean_value(
                    test.get("status", "Normal")
                )
            )
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        hide_index=True,
    )
    
########################################################
# Gemini Prompt
########################################################

def build_prompt(ocr_text: str) -> str:

    return f"""
You are an expert Pathologist and Medical Laboratory AI.

Analyze the OCR text from a laboratory report.

OCR TEXT
========
{ocr_text}

Return ONLY valid JSON.

Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanations.

Use this schema exactly:

{{
  "patient": {{
    "name": "",
    "age": "",
    "gender": ""
  }},
  "tests": [
    {{
      "name": "",
      "value": "",
      "unit": "",
      "reference_range": "",
      "status": "Normal"
    }}
  ],
  "summary": "",
  "recommendations": [
    ""
  ]
}}

Rules:

1. Extract every laboratory test.
2. Status must be:
   - Low
   - Normal
   - High
3. If patient information is unavailable leave it blank.
4. If no tests are detected return:

{{
  "patient": {{}},
  "tests": [],
  "summary": "No laboratory values detected.",
  "recommendations": []
}}

Return ONLY JSON.
"""


########################################################
# Gemini Analysis
########################################################

def analyze_lab_report(ocr_text):

    prompt = build_prompt(ocr_text)
    
    if not initialize_gemini():
        
     return {
        "patient": {},
        "tests": [],
        "summary": "Gemini initialization failed.",
        "recommendations": []
    }

    try:

        response = model.generate_content(prompt)

        if not hasattr(response, "text") or not response.text:
            raise ValueError("Gemini returned an empty response.")

        result = response.text.strip()

        # Remove markdown fences if present
        result = result.replace("```json", "")
        result = result.replace("```", "").strip()

        # Extract JSON object
        start = result.find("{")
        end = result.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in Gemini response.")

        json_text = result[start:end + 1]

        data = json.loads(json_text)

        # Ensure required keys exist
        data.setdefault("patient", {})
        data.setdefault("tests", [])
        data.setdefault("summary", "")
        data.setdefault("recommendations", [])

        return data

    except json.JSONDecodeError:

        st.error("Gemini returned invalid JSON.")

        return {
            "patient": {},
            "tests": [],
            "summary": "Unable to parse AI response.",
            "recommendations": []
        }

    except Exception as e:

        st.error(f"Gemini Error: {e}")

        return {
            "patient": {},
            "tests": [],
            "summary": "Analysis failed.",
            "recommendations": []
        }


########################################################
# Processing Pipeline
########################################################

def process_lab_report(image):

    ocr_text = extract_text(image)

    if not ocr_text:

        return {
            "ocr_text": "",
            "analysis": {
                "patient": {},
                "tests": [],
                "summary": "No text detected from the uploaded report.",
                "recommendations": []
            }
        }

    analysis = analyze_lab_report(ocr_text)

    return {
        "ocr_text": ocr_text,
        "analysis": analysis
    }


########################################################
# Summary Helpers
########################################################

def count_status(tests):

    counts = {
        "High": 0,
        "Low": 0,
        "Normal": 0
    }

    for test in tests:

        status = test.get("status", "Normal").capitalize()

        if status in counts:
            counts[status] += 1

    return counts


def show_summary(summary, recommendations):

    st.subheader("Medical Summary")

    if summary:
        st.info(summary)
    else:
        st.info("No summary available.")

    st.subheader("Recommendations")

    if recommendations:
        for item in recommendations:
            st.write(f"• {item}")
    else:
        st.write("No recommendations available.")
        

########################################################
# Main Streamlit Tab
########################################################

def lab_report_tab():

    st.header("🩺 AI Lab Report Analyzer")

    st.write(
        "Upload a laboratory report image for AI-powered analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload Lab Report",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Report",
        
    )

    if not st.button(
        "Analyze Report",
        type="primary"
    ):
        return

    with st.spinner("Reading laboratory report..."):
        result = process_lab_report(image)

    st.session_state["report_text"] = result["ocr_text"]
    st.session_state["lab_analysis"] = result["analysis"]

    analysis = result["analysis"]
    ocr_text = result["ocr_text"]

    st.success("Analysis Completed Successfully")

    ####################################################
    # OCR TEXT
    ####################################################

    with st.expander("View OCR Text"):

        st.text_area(
            "Extracted Text",
            value=ocr_text,
            height=250
        )

    ####################################################
    # PATIENT
    ####################################################

    patient = analysis.get("patient", {})

    show_patient(patient)

    ####################################################
    # TESTS
    ####################################################

    tests = analysis.get("tests", [])

    show_tests(tests)

    ####################################################
    # QUICK STATISTICS
    ####################################################

    if tests:

        counts = count_status(tests)

        st.subheader("Result Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🟢 Normal",
            counts["Normal"]
        )

        c2.metric(
            "🟠 Low",
            counts["Low"]
        )

        c3.metric(
            "🔴 High",
            counts["High"]
        )

    ####################################################
    # SUMMARY
    ####################################################

    show_summary(
        analysis.get("summary", ""),
        analysis.get("recommendations", [])
    )

    ####################################################
    # DOWNLOAD JSON
    ####################################################

    st.download_button(
        label="📥 Download Analysis (JSON)",
        data=json.dumps(
            analysis,
            indent=4
        ),
        file_name="lab_report_analysis.json",
        mime="application/json"
    )

    ####################################################
    # RAW JSON
    ####################################################

    with st.expander("View Raw JSON"):

        st.json(analysis)


########################################################
# Compatibility Wrapper
########################################################

def run():
    lab_report_tab()


########################################################
# Standalone Execution
########################################################

if __name__ == "__main__":
    lab_report_tab()