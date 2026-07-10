from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from PIL import Image
import io
import json
import base64
from pydantic import BaseModel
from typing import Optional
import platform
from dotenv import load_dotenv
import os

from backend.utils.ocr_engine import extract_text_safe

app = FastAPI(title="MedGuid AI Backend")

@app.get("/")
def root():
    return {"status": "ok", "message": "MedGuid AI Backend is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://medguid.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=API_KEY)

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
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        prompt = """You are a medical assistant. Analyze this prescription image.
Extract medicines and explain them.
Return ONLY a valid JSON array where each object has:
"name", "dosage", "frequency", "duration", "use"
No markdown, just pure JSON array."""
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}],
            max_tokens=4096
        )
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        return {"medicines": json.loads(raw_text.strip())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-lab-report")
async def analyze_lab_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = extract_text_safe(image)
        prompt = f"""You are a medical AI assistant. Analyze this lab report.
1. Extract test values and return JSON
2. Give a simple human language summary.
Return ONLY a valid JSON object:
{{"tests": [{{"name": "...", "value": "...", "status": "Normal" | "High" | "Low"}}], "summary": "..."}}
Lab Report text:
{text}"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        return {"report_text": text, "analysis": json.loads(raw_text.strip())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        prompt = f"""You are a friendly health assistant.
User question: {request.user_input}
Lab report: {request.report_text}
Instructions:
- Answer ONLY in {request.language}
- Use VERY simple language
- Max 2 lines only"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diet-recommendation")
async def diet_recommendation(request: DietRequest):
    try:
        if request.mode == "Quick Suggestions":
            prompt = f"""You are a nutrition expert. Based on this lab report, suggest a diet.
Keep it simple 4-5 lines, prefer Indian foods. Use bullet points.
Return JSON ONLY: {{"suggestions": ["point 1", "point 2"]}}
Lab Report: {request.report_text}"""
        else:
            prompt = f"""You are a professional nutritionist.
Create a personalized daily meal plan.
Diet: {request.diet_type}, Goal: {request.goal}, Meals: {request.meals}, Allergies: {request.allergies}
Medical Conditions from lab report: {request.report_text}
Return ONLY a valid JSON object:
{{"breakdown": [{{"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15}}], "plan": "text"}}"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        return json.loads(raw_text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
