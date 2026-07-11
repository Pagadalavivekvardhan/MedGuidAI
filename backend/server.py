from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from PIL import Image
import io
import json
import base64
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from dotenv import load_dotenv
import os

from backend.utils.ocr_engine import extract_text_safe
from backend.utils.image_preprocessing import enhance_for_vision_model
from backend.utils.prompts import PRESCRIPTION_TRANSCRIPTION_PROMPT, LAB_REPORT_ANALYSIS_PROMPT
from backend.security import setup_security, get_api_key, limiter


# --- Response Schemas ---


class Medicine(BaseModel):
    name: str = Field(..., description="Medicine name (exact spelling)")
    dosage: str = Field(..., description="Dosage (e.g., 500mg, 10mg)")
    frequency: str = Field(..., description="How often to take (e.g., Twice daily)")
    duration: str = Field(..., description="How long to take (e.g., 5 days)")
    use: str = Field(..., description="What it's for")
    instructions: str = Field(..., description="Special instructions")
    confidence: str = Field("unknown", description="Transcription confidence: high, medium, low, or unknown")
    raw_text: str = Field("", description="All other text visible near this medicine")


class PrescriptionResponse(BaseModel):
    medicines: List[Medicine] = Field(..., description="List of extracted medicines")


class TestResult(BaseModel):
    name: str = Field(..., description="Test name")
    value: str = Field(..., description="Numerical value")
    unit: str = Field(..., description="Unit (e.g., mg/dL, g/dL)")
    reference_range: str = Field(..., description="Normal range (e.g., 70-100)")
    status: str = Field(..., description="Normal, High, or Low")


class PatientInfo(BaseModel):
    name: str = Field("N/A", description="Patient name")
    age: str = Field("N/A", description="Patient age")
    gender: str = Field("N/A", description="Patient gender")


class LabAnalysis(BaseModel):
    patient: PatientInfo = Field(..., description="Patient information")
    tests: List[TestResult] = Field(..., description="List of test results")
    summary: str = Field(..., description="Medical summary in simple language")
    recommendations: List[str] = Field(..., description="Health recommendations")


class LabReportResponse(BaseModel):
    report_text: str = Field(..., description="OCR extracted text from the report")
    analysis: LabAnalysis = Field(..., description="Analysis results")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant's response")


class DietSuggestionResponse(BaseModel):
    suggestions: List[str] = Field(..., description="List of diet suggestions")


class MealBreakdown(BaseModel):
    meal: str = Field(..., description="Meal name (e.g., Breakfast)")
    calories: int = Field(..., description="Calories")
    carbs: int = Field(..., description="Carbohydrates in grams")
    protein: int = Field(..., description="Protein in grams")
    fat: int = Field(..., description="Fat in grams")


class DietPlanResponse(BaseModel):
    breakdown: List[MealBreakdown] = Field(..., description="Macronutrient breakdown per meal")
    plan: str = Field(..., description="Detailed meal plan text")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")


class RootResponse(BaseModel):
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Service message")


# --- App Configuration ---


app = FastAPI(
    title="MedGuid AI Backend",
    description="""Medical AI Assistant API for prescription analysis, lab report interpretation,
and diet recommendations powered by Groq and LLaMA vision models.""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

setup_security(app)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Health"],
    summary="Root endpoint",
)
@limiter.limit("30/minute")
async def root(request: Request):
    return {"status": "ok", "message": "MedGuid AI Backend is running"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
@limiter.limit("60/minute")
async def health(request: Request):
    return {"status": "healthy"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://medguid.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# Lazy Groq client initialization (no crash if key is missing at import time)
client = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def _get_groq_client():
    global client
    if client is None:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable is not set. Please configure it.",
            )
        client = Groq(api_key=GROQ_API_KEY)
    return client


class ChatRequest(BaseModel):
    user_input: str = Field(..., description="User's question")
    report_text: str = Field(..., description="Lab report text for context")
    language: str = Field(..., description="Response language (English, Hindi, Telugu)")


class DietRequest(BaseModel):
    report_text: str = Field(..., description="Lab report text for context")
    mode: str = Field(..., description="Diet mode: Quick Suggestions or Personalized Meal Plan")
    diet_type: Optional[str] = Field(None, description="Diet type for personalized mode")
    goal: Optional[str] = Field(None, description="Health goal")
    meals: Optional[int] = Field(None, description="Meals per day (3-5)")
    allergies: Optional[str] = Field(None, description="Any food allergies")


@app.post(
    "/api/upload-prescription",
    response_model=PrescriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        403: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Prescription"],
    summary="Extract medicines from prescription",
    description="""Upload a prescription image to extract medicine information using AI vision.

**Authentication:** Requires `X-API-KEY` header.
**Rate Limit:** 10 requests per minute per API key.

The API will:
1. Preprocess the image for better accuracy
2. Send to Groq vision model (LLaMA 4 Scout)
3. Extract all medicines with dosage, frequency, duration, and instructions

Supported formats: PNG, JPG, JPEG, WEBP""",
)
@limiter.limit("10/minute")
async def upload_prescription(request: Request, file: UploadFile = File(..., description="Prescription image file"), api_key: str = Depends(get_api_key)):
    try:
        groq_client = _get_groq_client()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        processed_image = enhance_for_vision_model(image)

        img_buffer = io.BytesIO()
        processed_image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        prompt = PRESCRIPTION_TRANSCRIPTION_PROMPT

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}],
            max_tokens=4096,
            temperature=0.1
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

        return {"medicines": json.loads(raw_text.strip())}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/analyze-lab-report",
    response_model=LabReportResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        403: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Lab Report"],
    summary="Analyze lab report",
    description="""Upload a lab report image to extract test results and get medical interpretation.

**Authentication:** Requires `X-API-KEY` header.
**Rate Limit:** 10 requests per minute per API key.

The API will:
1. Extract text using OCR (dual-engine: Tesseract + PaddleOCR)
2. Correct common OCR errors with RapidFuzz
3. Send OCR text to Groq LLaMA 3.3 for analysis
4. Return structured JSON with patient info, tests, and recommendations""",
)
@limiter.limit("10/minute")
async def analyze_lab_report(request: Request, file: UploadFile = File(..., description="Lab report image file"), api_key: str = Depends(get_api_key)):
    try:
        groq_client = _get_groq_client()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = extract_text_safe(image)

        prompt = LAB_REPORT_ANALYSIS_PROMPT.format(ocr_text=text)

        # Use text model for OCR text analysis (much cheaper than vision model)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in response.")

        analysis = json.loads(raw_text[start:end])

        analysis.setdefault("patient", {})
        analysis.setdefault("tests", [])
        analysis.setdefault("summary", "")
        analysis.setdefault("recommendations", [])

        for test in analysis.get("tests", []):
            status = test.get("status", "Normal")
            if status not in ["Normal", "High", "Low"]:
                test["status"] = "Normal"

        return {"report_text": text, "analysis": analysis}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Chat"],
    summary="Chat with health assistant",
    description="""Ask questions about your lab report in natural language.

**Authentication:** Requires `X-API-KEY` header.
**Rate Limit:** 20 requests per minute per API key.

The assistant will:
- Answer in your preferred language (English, Hindi, Telugu)
- Use simple, non-medical language
- Keep responses short and conversational""",
)
@limiter.limit("20/minute")
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    try:
        groq_client = _get_groq_client()
        prompt = f"""You are a friendly health assistant.

User question:
{chat_request.user_input}

Lab report:
{chat_request.report_text}

Instructions:
- Answer in {chat_request.language}
- Use VERY simple language
- Max 2 lines only
- Make it conversational
- Avoid medical jargon
- If needed, explain in simple words

Example style:
"Your hemoglobin is a bit low. This means your body may feel tired easily."

Keep it short and human-like."""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1
        )
        return {"reply": response.choices[0].message.content.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/diet-recommendation",
    response_model=Union[DietSuggestionResponse, DietPlanResponse],
    response_model_exclude_unset=True,
    responses={
        403: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Diet"],
    summary="Get diet recommendations",
    description="""Get personalized diet recommendations based on lab report.

**Authentication:** Requires `X-API-KEY` header.
**Rate Limit:** 10 requests per minute per API key.

Two modes:
- **Quick Suggestions**: Get 4-5 simple diet tips based on lab results
- **Personalized Meal Plan**: Get a full day meal plan with macro breakdown""",
)
@limiter.limit("10/minute")
async def diet_recommendation(request: Request, diet_request: DietRequest, api_key: str = Depends(get_api_key)):
    try:
        groq_client = _get_groq_client()
        if diet_request.mode == "Quick Suggestions":
            prompt = f"""You are a nutrition expert.

Based on this lab report, suggest a diet.

Rules:
- High sugar -> diabetic diet
- Low hemoglobin -> iron-rich foods
- High creatinine -> kidney-friendly foods
- Keep it simple in 4 to 5 lines
- Prefer Indian foods

Use bullet points.

Lab Report:
{diet_request.report_text}"""
        else:
            prompt = f"""You are a professional nutritionist.

Create a personalized daily meal plan.

User Details:
- Diet: {diet_request.diet_type}
- Goal: {diet_request.goal}
- Meals per day: {diet_request.meals}
- Allergies: {diet_request.allergies}

Medical Conditions (from lab report):
{diet_request.report_text}

Guidelines:
- Adjust diet based on medical conditions
- Suggest Indian foods
- Keep meals practical and realistic
- Return BOTH a detailed text plan AND a JSON formatted macro breakdown per meal.

Return EXACTLY in this format:

JSON:
[
  {{"meal": "Breakfast", "calories": 400, "carbs": 50, "protein": 20, "fat": 15}},
  {{"meal": "Lunch", "calories": 600, "carbs": 70, "protein": 35, "fat": 20}},
  {{"meal": "Dinner", "calories": 500, "carbs": 40, "protein": 40, "fat": 15}},
  {{"meal": "Snacks", "calories": 200, "carbs": 25, "protein": 5, "fat": 10}}
]

PLAN:
Your detailed meal plan schedule and recommendations."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        # For Quick Suggestions: return as suggestions object
        if diet_request.mode == "Quick Suggestions":
            suggestions = [line.strip().lstrip("-* ") for line in raw_text.split("\n") if line.strip()]
            return {"suggestions": suggestions}

        # For Personalized Meal Plan: extract JSON array and plan text
        try:
            if "JSON:" in raw_text and "PLAN:" in raw_text:
                json_part = raw_text.split("JSON:")[1].split("PLAN:")[0].strip()
                plan_part = raw_text.split("PLAN:")[1].strip()
            else:
                start = raw_text.find("[")
                end = raw_text.rfind("]") + 1
                if start != -1 and end > start:
                    json_part = raw_text[start:end]
                    plan_part = raw_text[:start].strip()
                else:
                    json_part = ""
                    plan_part = raw_text

            if "```json" in json_part:
                json_part = json_part.split("```json")[1].split("```")[0].strip()
            elif "```" in json_part:
                json_part = json_part.split("```")[1].split("```")[0].strip()

            breakdown = json.loads(json_part.strip()) if json_part else []
            return {"breakdown": breakdown, "plan": plan_part}
        except json.JSONDecodeError:
            return {"breakdown": [], "plan": raw_text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
