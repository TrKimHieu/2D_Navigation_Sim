@echo off
rem Turn .glb scenes into navigation maps and tasks (a dataset in data\datasets).
rem Usage: build-map.bat --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\run.ps1" build hm3denv build-map %*
exit /b %ERRORLEVEL%
