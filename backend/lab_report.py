import os
import json
import re
import base64
import io
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import pytesseract

import platform
from groq import Groq

from backend.utils.image_preprocessing import preprocess_image, preprocess_for_ocr
from backend.utils.ocr_engine import extract_text_dual
from backend.utils.text_correction import correct_ocr_text

client = None

def initialize_groq():
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.error("GROQ_API_KEY environment variable is not set.")
            return False
        client = Groq(api_key=api_key)
    return True

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR	esseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def extract_text(image):
    try:
        img_array = np.array(image)
        result = extract_text_dual(img_array)
        ocr_text = result["text"]
        corrected_text = correct_ocr_text(ocr_text)
        return corrected_text
    except Exception as e:
        st.warning(f"Dual OCR failed, falling back to Tesseract: {e}")
        processed = preprocess_image(image)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(processed, config=config)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

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
        return "High"
    if status == "low":
        return "Low"
    return "Normal"

def show_patient(patient):
    st.subheader("Patient Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("Name", patient.get("name", "N/A"))
    c2.metric("Age", patient.get("age", "N/A"))
    c3.metric("Gender", patient.get("gender", "N/A"))

def show_tests(tests):
    st.subheader("Laboratory Results")
    if len(tests) == 0:
        st.warning("No laboratory values detected.")
        return
    rows = []
    for test in tests:
        rows.append({
            "Test": clean_value(test.get("name")),
            "Value": clean_value(test.get("value")),
            "Unit": clean_value(test.get("unit")),
            "Reference": clean_value(test.get("reference_range")),
            "Status": status_color(clean_value(test.get("status", "Normal")))
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)

def build_prompt(ocr_text):
    return f"""
You are an expert Pathologist and Medical Laboratory AI.
Analyze the OCR text from a laboratory report.
OCR TEXT
========
{ocr_text}
Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use backticks.
Use this schema exactly:
{{
  "patient": {{"name": "", "age": "", "gender": ""}},
  "tests": [{{"name": "", "value": "", "unit": "", "reference_range": "", "status": "Normal"}}],
  "summary": "",
  "recommendations": [""]
}}
Rules:
1. Extract every laboratory test.
2. Status must be Low, Normal, or High.
3. If no tests detected return empty tests array.
Return ONLY JSON.
"""

def analyze_lab_report(ocr_text):
    prompt = build_prompt(ocr_text)
    if not initialize_groq():
        return {"patient": {}, "tests": [], "summary": "Groq initialization failed.", "recommendations": []}
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096
        )
        result = response.choices[0].message.content.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        start = result.find("{")
        end = result.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in Groq response.")
        data = json.loads(result[start:end + 1])
        data.setdefault("patient", {})
        data.setdefault("tests", [])
        data.setdefault("summary", "")
        data.setdefault("recommendations", [])
        return data
    except json.JSONDecodeError:
        st.error("Groq returned invalid JSON.")
        return {"patient": {}, "tests": [], "summary": "Unable to parse AI response.", "recommendations": []}
    except Exception as e:
        st.error(f"Groq Error: {e}")
        return {"patient": {}, "tests": [], "summary": "Analysis failed.", "recommendations": []}

def process_lab_report(image):
    ocr_text = extract_text(image)
    if not ocr_text:
        return {"ocr_text": "", "analysis": {"patient": {}, "tests": [], "summary": "No text detected.", "recommendations": []}}
    analysis = analyze_lab_report(ocr_text)
    return {"ocr_text": ocr_text, "analysis": analysis}

def count_status(tests):
    counts = {"High": 0, "Low": 0, "Normal": 0}
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
            st.write(f"* {item}")
    else:
        st.write("No recommendations available.")

def lab_report_tab():
    st.header("AI Lab Report Analyzer")
    st.write("Upload a laboratory report image for AI-powered analysis.")
    uploaded_file = st.file_uploader("Upload Lab Report", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        return
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Report")
    if not st.button("Analyze Report", type="primary"):
        return
    with st.spinner("Reading laboratory report..."):
        result = process_lab_report(image)
    st.session_state["report_text"] = result["ocr_text"]
    st.session_state["lab_analysis"] = result["analysis"]
    analysis = result["analysis"]
    ocr_text = result["ocr_text"]
    st.success("Analysis Completed Successfully")
    with st.expander("View OCR Text"):
        st.text_area("Extracted Text", value=ocr_text, height=250)
    patient = analysis.get("patient", {})
    show_patient(patient)
    tests = analysis.get("tests", [])
    show_tests(tests)
    if tests:
        counts = count_status(tests)
        st.subheader("Result Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal", counts["Normal"])
        c2.metric("Low", counts["Low"])
        c3.metric("High", counts["High"])
    show_summary(analysis.get("summary", ""), analysis.get("recommendations", []))
    st.download_button(label="Download Analysis (JSON)", data=json.dumps(analysis, indent=4), file_name="lab_report_analysis.json", mime="application/json")
    with st.expander("View Raw JSON"):
        st.json(analysis)

def run():
    lab_report_tab()

if __name__ == "__main__":
    lab_report_tab()
