@echo off
echo ========================================
echo   Retail vs Wholesale Analytics
echo   Starting Server...
echo ========================================
echo.

cd /d "%~dp0"
python analytics_dashboard.py

pause
