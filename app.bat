@echo off
rem Menu of every hm3denv feature (double-click me).
rem Usage: app.bat --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\run.ps1" - scripts\app.py %*
exit /b %ERRORLEVEL%
