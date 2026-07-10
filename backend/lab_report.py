import os
import json
import re

import pandas as pd
from PIL import Image

import streamlit as st

from groq import Groq

import pytesseract
from backend.utils.ocr_engine import extract_text_safe
from backend.utils.image_preprocessing import preprocess_image
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

def extract_text(image):
    try:
        result = extract_text_safe(image)
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
            "Status": clean_value(test.get("status", "Normal")).capitalize()
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)

def build_prompt(ocr_text):
    return f"""You are an expert pathologist and medical laboratory AI assistant.

Analyze the following OCR-extracted text from a medical laboratory report.
Your task is to extract ALL test results accurately and provide a medical interpretation.

OCR EXTRACTED TEXT:
==================
{ocr_text}
==================

Return ONLY a valid JSON object with this EXACT structure:
{{
  "patient": {{"name": "string or N/A", "age": "string or N/A", "gender": "string or N/A"}},
  "tests": [
    {{
      "name": "Test name",
      "value": "Numerical value",
      "unit": "Unit (e.g., mg/dL, g/dL, %)",
      "reference_range": "Normal range (e.g., 70-100)",
      "status": "Normal" | "High" | "Low"
    }}
  ],
  "summary": "Brief medical summary of findings in simple language",
  "recommendations": ["Recommendation 1", "Recommendation 2"]
}}

CRITICAL RULES:
1. Extract EVERY single test result visible in the text
2. Compare each value against its reference range to determine status
3. Status must be exactly one of: "Normal", "High", or "Low"
4. If a test value is above the upper limit of reference range, status = "High"
5. If a test value is below the lower limit of reference range, status = "Low"
6. If within range or no reference range, status = "Normal"
7. The summary should explain findings in simple, non-medical language
8. Recommendations should be practical health advice
9. Return ONLY the JSON object - no markdown, no backticks, no extra text
10. If no tests are found, return empty tests array with an appropriate message in summary

Return ONLY the JSON object."""

def analyze_lab_report(ocr_text):
    prompt = build_prompt(ocr_text)
    if not initialize_groq():
        return {"patient": {}, "tests": [], "summary": "Groq initialization failed.", "recommendations": []}
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting
        if "```json" in result:
            result = result.split("```json")[1]
        if "```" in result:
            result = result.split("```")[0]
        
        # Extract JSON object
        start = result.find("{")
        end = result.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in Groq response.")
        
        data = json.loads(result[start:end])
        
        # Ensure all required fields exist with defaults
        data.setdefault("patient", {})
        data.setdefault("tests", [])
        data.setdefault("summary", "")
        data.setdefault("recommendations", [])
        
        # Validate and normalize test statuses
        for test in data.get("tests", []):
            status = test.get("status", "Normal")
            if status not in ["Normal", "High", "Low"]:
                test["status"] = "Normal"
        
        return data
    except json.JSONDecodeError:
        st.error("Groq returned invalid JSON. Please try again.")
        return {"patient": {}, "tests": [], "summary": "Unable to parse AI response. Please try uploading the image again.", "recommendations": []}
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return {"patient": {}, "tests": [], "summary": "Analysis failed. Please ensure the image contains readable text.", "recommendations": []}

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
    st.image(image, caption="Uploaded Report", use_container_width=True)
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


