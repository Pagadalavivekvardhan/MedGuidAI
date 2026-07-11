import os
import json
import re
import io
from datetime import datetime

import pandas as pd
from PIL import Image

import streamlit as st

# Try to import API client for backend mode
try:
    from frontend.api_client import analyze_lab_report as api_analyze_lab_report
    from frontend.api_client import get_api_key, check_backend_health
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False

# Import direct Groq client as fallback
client = None
try:
    from groq import Groq
except ImportError:
    pass

try:
    import pytesseract
except ImportError:
    pytesseract = None
from backend.utils.ocr_engine import extract_text_safe
from backend.utils.image_preprocessing import preprocess_image
from backend.utils.text_correction import correct_ocr_text

# Configurable reports directory
REPORTS_DIR = os.getenv("REPORTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_reports"))


def initialize_groq():
    global client
    if client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return False
        try:
            client = Groq(api_key=api_key)
        except Exception:
            return False
    return True


def extract_text(image):
    """Extract text from image using OCR pipeline.

    extract_text_safe returns a string directly (not a dict).
    """
    try:
        ocr_text = extract_text_safe(image)
        if not ocr_text or not ocr_text.strip():
            raise ValueError("OCR returned empty text")
        corrected_text = correct_ocr_text(ocr_text)
        return corrected_text
    except Exception as e:
        if pytesseract is None:
            st.error("pytesseract not available for fallback OCR.")
            return ""
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
        # Use text model for OCR text analysis (not vision model)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()

        if "```json" in result:
            result = result.split("```json")[1]
        if "```" in result:
            result = result.split("```")[0]

        start = result.find("{")
        end = result.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in Groq response.")

        data = json.loads(result[start:end])

        data.setdefault("patient", {})
        data.setdefault("tests", [])
        data.setdefault("summary", "")
        data.setdefault("recommendations", [])

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


def _process_via_api(image):
    """Process lab report using the FastAPI backend with API key authentication."""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    result = api_analyze_lab_report(img_buffer.getvalue(), "lab_report.png")
    return {"ocr_text": result.get("report_text", ""), "analysis": result.get("analysis", {})}


def _process_via_direct(image):
    """Process lab report using direct Groq API call (fallback)."""
    ocr_text = extract_text(image)
    if not ocr_text:
        return {"ocr_text": "", "analysis": {"patient": {}, "tests": [], "summary": "No text detected.", "recommendations": []}}
    analysis_result = analyze_lab_report(ocr_text)
    return {"ocr_text": ocr_text, "analysis": analysis_result}


def process_lab_report(image):
    """Process lab report.

    Uses API backend with API key if configured, otherwise falls back to direct Groq calls.
    """
    use_api = (
        HAS_API_CLIENT
        and get_api_key()
        and check_backend_health()
    )

    try:
        if use_api:
            return _process_via_api(image)
        else:
            return _process_via_direct(image)
    except Exception as e:
        st.error(f"Error processing lab report: {e}")
        if use_api:
            st.info("Falling back to direct mode...")
            try:
                return _process_via_direct(image)
            except Exception as e2:
                st.error(f"Direct mode also failed: {e2}")
        return {"ocr_text": "", "analysis": {"patient": {}, "tests": [], "summary": str(e), "recommendations": []}}


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


# ─── Disk Persistence ───────────────────────────────────────────────────────


def _save_report_to_disk(analysis, ocr_text, filename_prefix="lab_report"):
    """Save lab report analysis to disk for persistence.

    Returns the saved file path, or None if saving failed.
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(REPORTS_DIR, filename)

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "ocr_text": ocr_text,
            "analysis": analysis,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

        return filepath
    except Exception as e:
        st.warning(f"Could not save report to disk: {e}")
        return None


def _load_saved_reports():
    """Load all saved reports from disk, sorted by timestamp (newest first).

    Returns a list of dicts with filename, timestamp, and summary info.
    """
    reports = []
    if not os.path.exists(REPORTS_DIR):
        return reports

    for filename in sorted(os.listdir(REPORTS_DIR), reverse=True):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(REPORTS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            reports.append({
                "filename": filename,
                "filepath": filepath,
                "timestamp": data.get("timestamp", "Unknown"),
                "patient": data.get("analysis", {}).get("patient", {}),
                "tests_count": len(data.get("analysis", {}).get("tests", [])),
                "summary": data.get("analysis", {}).get("summary", "No summary"),
            })
        except Exception:
            continue
    return reports


def _display_saved_reports():
    """Display the Previous Reports section."""
    reports = _load_saved_reports()
    if not reports:
        st.info("No previous reports saved yet.")
        return

    st.write(f"**{len(reports)} saved report(s)**")
    for i, report in enumerate(reports[:10]):  # Show latest 10
        patient_name = report["patient"].get("name", "Unknown")
        ts = report["timestamp"][:19].replace("T", " ") if report["timestamp"] != "Unknown" else "Unknown"

        with st.expander(f"{patient_name} - {ts}"):
            st.caption(f"File: {report['filename']}")
            st.caption(f"Tests: {report['tests_count']}")
            st.write(report["summary"][:200] + "..." if len(report["summary"]) > 200 else report["summary"])

            if st.button("Load Report", key=f"load_saved_{i}"):
                try:
                    with open(report["filepath"], "r", encoding="utf-8") as f:
                        data = json.load(f)
                    st.session_state["report_text"] = data.get("ocr_text", "")
                    st.session_state["lab_analysis"] = data.get("analysis", {})
                    st.session_state["_lab_loaded_from_disk"] = True
                    st.success("Report loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load report: {e}")


# ─── Main Tab ───────────────────────────────────────────────────────────────


def lab_report_tab():
    st.header("AI Lab Report Analyzer")

    # Check if a saved report was loaded
    if st.session_state.get("_lab_loaded_from_disk"):
        analysis = st.session_state["lab_analysis"]
        ocr_text = st.session_state.get("report_text", "")

        if st.button("New Analysis", type="secondary"):
            st.session_state.pop("lab_analysis", None)
            st.session_state.pop("report_text", None)
            st.session_state.pop("_lab_loaded_from_disk", None)
            st.rerun()

        st.success("Report loaded from saved reports")
        with st.expander("View OCR Text"):
            st.text_area("Extracted Text", value=ocr_text, height=250, disabled=True)
        show_patient(analysis.get("patient", {}))
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
        return

    # Show Previous Reports section
    with st.expander("Previous Reports", expanded=False):
        _display_saved_reports()

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

    # Save report to disk automatically
    saved_path = _save_report_to_disk(analysis, ocr_text)
    if saved_path:
        st.info(f"Report saved to: {os.path.basename(saved_path)}")

    with st.expander("View OCR Text"):
        st.text_area("Extracted Text", value=ocr_text, height=250)
    show_patient(analysis.get("patient", {}))
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
