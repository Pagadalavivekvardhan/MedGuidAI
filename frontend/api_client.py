"""API client for making HTTP requests to the MedGuid FastAPI backend.

All requests include the X-API-KEY header for authentication.
"""

import requests
import streamlit as st
from typing import Optional, Dict, Any, List


def get_backend_url() -> str:
    """Get the backend URL from session state or default."""
    return st.session_state.get("backend_url", "http://localhost:8000")


def get_api_key() -> Optional[str]:
    """Get the API key from session state."""
    return st.session_state.get("api_key", "")


def get_headers() -> Dict[str, str]:
    """Get request headers with API key."""
    headers = {"Content-Type": "application/json"}
    api_key = get_api_key()
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


def check_backend_health() -> bool:
    """Check if the backend is running and reachable."""
    try:
        response = requests.get(
            f"{get_backend_url()}/health",
            headers=get_headers(),
            timeout=5
        )
        return response.status_code == 200
    except (requests.exceptions.RequestException, ConnectionError, OSError):
        return False


def upload_prescription(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload a prescription image and get extracted medicines.
    
    Args:
        file_bytes: The image file bytes
        filename: The original filename
        
    Returns:
        Dict with 'medicines' key containing list of medicines
        
    Raises:
        Exception: If the API call fails
    """
    headers = get_headers()
    # Remove Content-Type for file upload (requests sets it automatically with multipart)
    if "Content-Type" in headers:
        del headers["Content-Type"]
    
    files = {"file": (filename, file_bytes, "image/png")}
    
    try:
        response = requests.post(
            f"{get_backend_url()}/api/upload-prescription",
            headers=headers,
            files=files,
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Cannot connect to backend: {e}")
    
    if response.status_code == 403:
        raise Exception("Invalid or missing API key. Please check your API key in the sidebar.")
    elif response.status_code == 429:
        raise Exception("Rate limit exceeded. Please wait before making another request.")
    elif response.status_code != 200:
        detail = response.json().get("detail", "Unknown error") if response.text else "Unknown error"
        raise Exception(f"API error: {detail}")
    
    return response.json()


def analyze_lab_report(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Upload a lab report image and get analysis.
    
    Args:
        file_bytes: The image file bytes
        filename: The original filename
        
    Returns:
        Dict with 'report_text' and 'analysis' keys
        
    Raises:
        Exception: If the API call fails
    """
    headers = get_headers()
    if "Content-Type" in headers:
        del headers["Content-Type"]
    
    files = {"file": (filename, file_bytes, "image/png")}
    
    try:
        response = requests.post(
            f"{get_backend_url()}/api/analyze-lab-report",
            headers=headers,
            files=files,
            timeout=60
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Cannot connect to backend: {e}")
    
    if response.status_code == 403:
        raise Exception("Invalid or missing API key. Please check your API key in the sidebar.")
    elif response.status_code == 429:
        raise Exception("Rate limit exceeded. Please wait before making another request.")
    elif response.status_code != 200:
        detail = response.json().get("detail", "Unknown error") if response.text else "Unknown error"
        raise Exception(f"API error: {detail}")
    
    return response.json()


def send_chat_message(user_input: str, report_text: str, language: str) -> str:
    """Send a chat message and get response.
    
    Args:
        user_input: The user's question
        report_text: The lab report text for context
        language: Response language (English, Hindi, Telugu)
        
    Returns:
        The assistant's response text
        
    Raises:
        Exception: If the API call fails
    """
    headers = get_headers()
    
    payload = {
        "user_input": user_input,
        "report_text": report_text,
        "language": language
    }
    
    try:
        response = requests.post(
            f"{get_backend_url()}/api/chat",
            headers=headers,
            json=payload,
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Cannot connect to backend: {e}")
    
    if response.status_code == 403:
        raise Exception("Invalid or missing API key. Please check your API key in the sidebar.")
    elif response.status_code == 429:
        raise Exception("Rate limit exceeded. Please wait before making another request.")
    elif response.status_code != 200:
        detail = response.json().get("detail", "Unknown error") if response.text else "Unknown error"
        raise Exception(f"API error: {detail}")
    
    return response.json().get("reply", "")


def get_diet_quick_suggestions(report_text: str) -> List[str]:
    """Get quick diet suggestions based on lab report.
    
    Args:
        report_text: The lab report text for context
        
    Returns:
        List of diet suggestion strings
        
    Raises:
        Exception: If the API call fails
    """
    headers = get_headers()
    
    payload = {
        "report_text": report_text,
        "mode": "Quick Suggestions"
    }
    
    try:
        response = requests.post(
            f"{get_backend_url()}/api/diet-recommendation",
            headers=headers,
            json=payload,
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Cannot connect to backend: {e}")
    
    if response.status_code == 403:
        raise Exception("Invalid or missing API key. Please check your API key in the sidebar.")
    elif response.status_code == 429:
        raise Exception("Rate limit exceeded. Please wait before making another request.")
    elif response.status_code != 200:
        detail = response.json().get("detail", "Unknown error") if response.text else "Unknown error"
        raise Exception(f"API error: {detail}")
    
    data = response.json()
    return data.get("suggestions", [])


def get_diet_personalized_plan(
    report_text: str,
    diet_type: str,
    goal: str,
    meals: int,
    allergies: str
) -> Dict[str, Any]:
    """Get personalized diet meal plan.
    
    Args:
        report_text: The lab report text for context
        diet_type: Vegetarian or Non-Vegetarian
        goal: General Health, Weight Loss, or Muscle Gain
        meals: Number of meals per day (3-5)
        allergies: Any food allergies
        
    Returns:
        Dict with 'breakdown' (list of meals) and 'plan' (text) keys
        
    Raises:
        Exception: If the API call fails
    """
    headers = get_headers()
    
    payload = {
        "report_text": report_text,
        "mode": "Personalized Meal Plan",
        "diet_type": diet_type,
        "goal": goal,
        "meals": meals,
        "allergies": allergies or ""
    }
    
    try:
        response = requests.post(
            f"{get_backend_url()}/api/diet-recommendation",
            headers=headers,
            json=payload,
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Cannot connect to backend: {e}")
    
    if response.status_code == 403:
        raise Exception("Invalid or missing API key. Please check your API key in the sidebar.")
    elif response.status_code == 429:
        raise Exception("Rate limit exceeded. Please wait before making another request.")
    elif response.status_code != 200:
        detail = response.json().get("detail", "Unknown error") if response.text else "Unknown error"
        raise Exception(f"API error: {detail}")
    
    return response.json()
