@echo off
rem Download pre-built datasets from Hugging Face into data\datasets.
rem Usage: get-data.bat --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\run.ps1" hub scripts\get_data.py %*
exit /b %ERRORLEVEL%
