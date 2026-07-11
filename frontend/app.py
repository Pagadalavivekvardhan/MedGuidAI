import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from frontend.api_client import (
    check_backend_health,
    get_backend_url,
    get_api_key,
)
from backend.prescription import prescription_tab
from backend.lab_report import lab_report_tab, get_saved_reports_count
from backend.diet_recommendation import diet_tab
from backend.chat_assistant import chat_tab

st.set_page_config(page_title="Medical AI Agent", layout="wide")

st.markdown("""
    <style>
    /* Medical Theme CSS */
    .stApp {
        background-color: #F5C9E8;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #2e6c80;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);            

        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e4a5a;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        color: white;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: API Configuration ---

with st.sidebar:
    st.header("API Configuration")

    # Backend URL
    backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.get("backend_url", "http://localhost:8000"),
        help="URL of the MedGuid FastAPI backend"
    )
    st.session_state["backend_url"] = backend_url

    # API Key
    api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        help="Your MedGuid API key (X-API-KEY header)"
    )
    st.session_state["api_key"] = api_key

    # Connection status
    if st.button("Check Connection"):
        if check_backend_health():
            st.success("Backend is running!")
        else:
            st.error("Cannot reach backend")

    # Show saved reports count (lightweight - no JSON parsing)
    saved_count = get_saved_reports_count()
    if saved_count > 0:
        st.success(f"📄 {saved_count} saved report(s)")
    else:
        st.info("No saved reports yet")

    # Show current config
    st.divider()
    st.caption(f"Backend: `{get_backend_url()}`")
    st.caption(f"API Key: `{'***' + api_key[-4:] if api_key and len(api_key) >= 4 else 'Not set'}`")

# --- Main Content ---

st.title("Medical AI Agent")
tab1, tab2, tab3, tab4 = st.tabs([
    "Prescription Reader",   
    "Lab Report Analyzer",
    "Diet Recommendation",
    "Chat Assistant"
])

with tab1:
    prescription_tab()

with tab2:
    lab_report_tab()

with tab3:
    diet_tab()

with tab4:
    chat_tab()