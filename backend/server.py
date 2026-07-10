from fastapi import FastAPI, UploadFile, File, HTTPException
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


# ─── Response Schemas ───────────────────────────────────────────────────────


class Medicine(BaseModel):
    """Schema for a single extracted medicine."""
    name: str = Field(..., description="Medicine name (exact spelling)")
    dosage: str = Field(..., description="Dosage (e.g., 500mg, 10mg)")
    frequency: str = Field(..., description="How often to take (e.g., Twice daily)")
    duration: str = Field(..., description="How long to take (e.g., 5 days)")
    use: str = Field(..., description="What it's for")
    instructions: str = Field(..., description="Special instructions")


class PrescriptionResponse(BaseModel):
    """Response schema for prescription extraction."""
    medicines: List[Medicine] = Field(..., description="List of extracted medicines")


class TestResult(BaseModel):
    """Schema for a single lab test result."""
    name: str = Field(..., description="Test name")
    value: str = Field(..., description="Numerical value")
    unit: str = Field(..., description="Unit (e.g., mg/dL, g/dL)")
    reference_range: str = Field(..., description="Normal range (e.g., 70-100)")
    status: str = Field(..., description="Normal, High, or Low")


class PatientInfo(BaseModel):
    """Schema for patient information."""
    name: str = Field("N/A", description="Patient name")
    age: str = Field("N/A", description="Patient age")
    gender: str = Field("N/A", description="Patient gender")


class LabAnalysis(BaseModel):
    """Schema for lab report analysis."""
    patient: PatientInfo = Field(..., description="Patient information")
    tests: List[TestResult] = Field(..., description="List of test results")
    summary: str = Field(..., description="Medical summary in simple language")
    recommendations: List[str] = Field(..., description="Health recommendations")


class LabReportResponse(BaseModel):
    """Response schema for lab report analysis."""
    report_text: str = Field(..., description="OCR extracted text from the report")
    analysis: LabAnalysis = Field(..., description="Analysis results")


class ChatResponse(BaseModel):
    """Response schema for chat assistant."""
    reply: str = Field(..., description="Assistant's response")


class DietSuggestionResponse(BaseModel):
    """Response schema for quick diet suggestions."""
    suggestions: List[str] = Field(..., description="List of diet suggestions")


class MealBreakdown(BaseModel):
    """Schema for a single meal's macro breakdown."""
    meal: str = Field(..., description="Meal name (e.g., Breakfast)")
    calories: int = Field(..., description="Calories")
    carbs: int = Field(..., description="Carbohydrates in grams")
    protein: int = Field(..., description="Protein in grams")
    fat: int = Field(..., description="Fat in grams")


class DietPlanResponse(BaseModel):
    """Response schema for personalized meal plan."""
    breakdown: List[MealBreakdown] = Field(..., description="Macronutrient breakdown per meal")
    plan: str = Field(..., description="Detailed meal plan text")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Health status")


class RootResponse(BaseModel):
    """Schema for root endpoint response."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Service message")


# ─── App Configuration ──────────────────────────────────────────────────────


app = FastAPI(
    title="MedGuid AI Backend",
    description="""Medical AI Assistant API for prescription analysis, lab report interpretation,
and diet recommendations powered by Groq and LLaMA vision models.""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.get(
    "/",
    response_model=RootResponse,
    tags=["Health"],
    summary="Root endpoint",
    description="Returns the API status and a welcome message.",
)
def root():
    return {"status": "ok", "message": "MedGuid AI Backend is running"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns the health status of the API.",
)
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

class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    user_input: str = Field(..., description="User's question")
    report_text: str = Field(..., description="Lab report text for context")
    language: str = Field(..., description="Response language (English, Hindi, Telugu)")


class DietRequest(BaseModel):
    """Request schema for diet recommendation endpoint."""
    report_text: str = Field(..., description="Lab report text for context")
    mode: str = Field(..., description="Diet mode: Quick Suggestions or Personalized Meal Plan")
    diet_type: Optional[str] = Field(None, description="Diet type for personalized mode (Vegetarian or Non-Vegetarian)")
    goal: Optional[str] = Field(None, description="Health goal (General Health, Weight Loss, Muscle Gain)")
    meals: Optional[int] = Field(None, description="Meals per day (3-5)")
    allergies: Optional[str] = Field(None, description="Any food allergies")

@app.post(
    "/api/upload-prescription",
    response_model=PrescriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Prescription"],
    summary="Extract medicines from prescription",
    description="""Upload a prescription image to extract medicine information using AI vision.

The API will:
1. Preprocess the image for better accuracy
2. Send to Groq vision model (LLaMA 4 Scout)
3. Extract all medicines with dosage, frequency, duration, and instructions

Supported formats: PNG, JPG, JPEG, WEBP""",
)
async def upload_prescription(file: UploadFile = File(..., description="Prescription image file")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess image for better accuracy
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

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}],
            max_tokens=4096,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()

        # Clean up markdown formatting
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        # Extract JSON array
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        if start != -1 and end > start:
            raw_text = raw_text[start:end]

        return {"medicines": json.loads(raw_text.strip())}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/api/analyze-lab-report",
    response_model=LabReportResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Lab Report"],
    summary="Analyze lab report",
    description="""Upload a lab report image to extract test results and get medical interpretation.

The API will:
1. Extract text using OCR (dual-engine: Tesseract + PaddleOCR)
2. Correct common OCR errors with RapidFuzz
3. Send to Groq vision model for analysis
4. Return structured JSON with patient info, tests, and recommendations""",
)
async def analyze_lab_report(file: UploadFile = File(..., description="Lab report image file")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = extract_text_safe(image)

        prompt = f"""You are an expert pathologist and medical laboratory AI assistant.

Analyze the following OCR-extracted text from a medical laboratory report.
Your task is to extract ALL test results accurately and provide a medical interpretation.

OCR EXTRACTED TEXT:
==================
{text}
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

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()

        # Clean up markdown formatting
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        # Extract JSON object
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in response.")

        analysis = json.loads(raw_text[start:end])

        # Ensure all required fields exist
        analysis.setdefault("patient", {})
        analysis.setdefault("tests", [])
        analysis.setdefault("summary", "")
        analysis.setdefault("recommendations", [])

        # Validate and normalize test statuses
        for test in analysis.get("tests", []):
            status = test.get("status", "Normal")
            if status not in ["Normal", "High", "Low"]:
                test["status"] = "Normal"

        return {"report_text": text, "analysis": analysis}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Chat"],
    summary="Chat with health assistant",
    description="""Ask questions about your lab report in natural language.

The assistant will:
- Answer in your preferred language (English, Hindi, Telugu)
- Use simple, non-medical language
- Keep responses short and conversational""",
)
async def chat(request: ChatRequest):
    try:
        prompt = f"""You are a friendly health assistant.

User question:
{request.user_input}

Lab report:
{request.report_text}

Instructions:
- Answer in {request.language}
- Use VERY simple language
- Max 2 lines only
- Make it conversational
- Avoid medical jargon
- If needed, explain in simple words

Example style:
"Your hemoglobin is a bit low. This means your body may feel tired easily."

Keep it short and human-like."""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/api/diet-recommendation",
    response_model=Union[DietSuggestionResponse, DietPlanResponse],
    responses={
        500: {"model": ErrorResponse, "description": "AI processing failed"},
    },
    tags=["Diet"],
    summary="Get diet recommendations",
    description="""Get personalized diet recommendations based on lab report.

Two modes:
- **Quick Suggestions**: Get 4-5 simple diet tips based on lab results
- **Personalized Meal Plan**: Get a full day meal plan with macro breakdown""",
)
async def diet_recommendation(request: DietRequest):
    try:
        if request.mode == "Quick Suggestions":
            prompt = f"""You are a nutrition expert.

Based on this lab report, suggest a diet.

Rules:
- High sugar \u2192 diabetic diet
- Low hemoglobin \u2192 iron-rich foods
- High creatinine \u2192 kidney-friendly foods
- Keep it simple in 4 to 5 lines
- Prefer Indian foods

Use bullet points.

Lab Report:
{request.report_text}"""
        else:
            prompt = f"""You are a professional nutritionist.

Create a personalized daily meal plan.

User Details:
- Diet: {request.diet_type}
- Goal: {request.goal}
- Meals per day: {request.meals}
- Allergies: {request.allergies}

Medical Conditions (from lab report):
{request.report_text}

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

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()

        # Clean up markdown formatting
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1]
        if "```" in raw_text:
            raw_text = raw_text.split("```")[0]

        # For Quick Suggestions: return as suggestions object
        if request.mode == "Quick Suggestions":
            suggestions = [line.strip().lstrip("-* ") for line in raw_text.split("\n") if line.strip()]
            return {"suggestions": suggestions}

        # For Personalized Meal Plan: extract JSON array and plan text
        try:
            # Try to extract JSON array from JSON: PLAN: format
            if "JSON:" in raw_text and "PLAN:" in raw_text:
                json_part = raw_text.split("JSON:")[1].split("PLAN:")[0].strip()
                plan_part = raw_text.split("PLAN:")[1].strip()
            else:
                # Fallback: try to find JSON array directly
                start = raw_text.find("[")
                end = raw_text.rfind("]") + 1
                if start != -1 and end > start:
                    json_part = raw_text[start:end]
                    plan_part = raw_text[:start].strip()
                else:
                    json_part = ""
                    plan_part = raw_text

            # Clean markdown from JSON part
            if "```json" in json_part:
                json_part = json_part.split("```json")[1].split("```")[0].strip()
            elif "```" in json_part:
                json_part = json_part.split("```")[1].split("```")[0].strip()

            breakdown = json.loads(json_part.strip()) if json_part else []
            return {"breakdown": breakdown, "plan": plan_part}
        except json.JSONDecodeError:
            # Fallback: return raw text as plan
            return {"breakdown": [], "plan": raw_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
