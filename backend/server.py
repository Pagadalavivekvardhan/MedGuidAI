from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import io
import pytesseract
import cv2
import numpy as np
import json
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="MedGuid AI Backend")

@app.get("/")
def root():
    return {"status": "ok", "message": "MedGuid AI Backend is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:8501",
    "https://medguid.streamlit.app"
],  # TODO: Restrict to actual Streamlit Cloud URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import platform
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or Render dashboard.")

genai.configure(api_key=API_KEY)
# Using the same key from existing code, though a centralized location or env var is better.
# Make sure tesseract path is set correctly as in existing code
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

class ChatRequest(BaseModel):
    user_input: str
    report_text: str
    language: str

class DietRequest(BaseModel):
    report_text: str
    mode: str
    diet_type: Optional[str] = None
    goal: Optional[str] = None
    meals: Optional[int] = None
    allergies: Optional[str] = None


@app.post("/api/upload-prescription")
async def upload_prescription(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """
You are a medical assistant.
Analyze this prescription image.
Extract medicines and explain them.
Return ONLY a valid JSON array where each object has these keys:
"name", "dosage", "frequency", "duration", "use"
Example:
[
  {
    "name": "Paracetamol 500mg",
    "dosage": "1 tablet",
    "frequency": "twice daily",
    "duration": "5 days",
    "use": "For fever and pain relief"
  }
]
No markdown, just pure JSON array.
"""
        response = model.generate_content([prompt, image])
        # Clean response text if it contains markdown code blocks
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        
        medicines = json.loads(raw_text.strip())
        return {"medicines": medicines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-lab-report")
async def analyze_lab_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        text = pytesseract.image_to_string(thresh)

        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
You are a medical AI assistant. Analyze this lab report.
1. Extract test values and return JSON
2. Give a simple human language summary of health status.

Return ONLY a valid JSON object in this exact format:
{{
  "tests": [
    {{"name": "...", "value": "...", "status": "Normal" | "High" | "Low"}}
  ],
  "summary": "..."
}}

Lab Report text:
{text}
"""
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]

        data = json.loads(raw_text.strip())
        return {"report_text": text, "analysis": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
You are a friendly health assistant.

User question:
{request.user_input}

Lab report:
{request.report_text}

Instructions:
- Answer ONLY in {request.language}
- Use VERY simple language
- Max 2 lines only
- Make it conversational
- Avoid medical jargon
- If needed, explain in simple words
"""
        response = model.generate_content(prompt)
        return {"reply": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diet-recommendation")
async def diet_recommendation(request: DietRequest):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        if request.mode == "Quick Suggestions":
            prompt = f"""
You are a nutrition expert. Based on this lab report, suggest a diet.
Rules: High sugar -> diabetic diet, Low hemoglobin -> iron-rich foods, etc.
Keep it simple 4-5 lines, prefer Indian foods. Use bullet points.
Return JSON ONLY:
{{ "suggestions": ["point 1", "point 2"] }}

Lab Report: {request.report_text}
"""
        else:
            prompt = f"""
You are a professional nutritionist.
Create a personalized daily meal plan.
Diet: {request.diet_type}, Goal: {request.goal}, Meals: {request.meals}, Allergies: {request.allergies}
Medical Conditions from lab report: {request.report_text}

Return ONLY a valid JSON object with detailed text plan and a macro breakdown per meal:
{{
  "breakdown": [
    {{"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15}}
  ],
  "plan": "Complete text description going here..."
}}
"""
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
            
        data = json.loads(raw_text.strip())
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
