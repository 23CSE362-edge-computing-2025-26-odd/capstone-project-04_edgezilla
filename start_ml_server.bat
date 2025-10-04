@echo off
echo Starting Sepsis ML Server...
echo.
echo Installing dependencies...
pip install -r ml_requirements.txt
echo.
echo Starting server on http://localhost:5002
echo Press Ctrl+C to stop the server
echo.
python ml_server.py