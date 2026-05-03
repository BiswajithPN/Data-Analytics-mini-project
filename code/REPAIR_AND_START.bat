@echo off
title 🚀 ANALYTICS DASHBOARD REPAIR TOOL
echo ======================================================
echo 🔍 STEP 1: Detecting old server processes...
echo ======================================================

:: Find PID on port 5000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    echo 🔪 Found old server (PID %%a). Killing it now...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ======================================================
echo 🚀 STEP 2: Starting the NEW Stabilized Server (v2.0)
echo ======================================================
echo Please wait for the "VERSION 2.0" message to appear...
echo.

python analytics_dashboard.py

pause
