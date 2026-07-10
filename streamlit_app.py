"""
MedGuid AI - Streamlit Entry Point
For Streamlit Cloud, point the app to this file OR frontend/app.py directly.
"""
import subprocess
import sys
import os

app_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
