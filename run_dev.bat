@echo off
echo Starting FastAPI Backend...
start cmd /k ".\venv\Scripts\python.exe -m uvicorn server:app --reload --port 8000"

echo Starting React Frontend...
cd frontend
start cmd /k "npm run dev"

echo Both servers are starting in new windows!
pause
