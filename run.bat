@echo off
echo Installing packages (first time only)...
pip install -r requirements.txt
echo.
echo Starting Zero Complaint System at http://localhost:8000
python server.py
pause
