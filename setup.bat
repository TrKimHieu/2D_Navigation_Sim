@echo off
rem Install hm3denv into .venv. Options: --hub --build --sim --train --all (setup.bat --help)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\setup.ps1" %*
exit /b %ERRORLEVEL%
