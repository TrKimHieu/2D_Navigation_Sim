@echo off
rem Continue a dataset: new scenes, more tasks on its maps, new robots.
rem Usage: extend-data.bat --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\run.ps1" build,hub hm3denv extend %*
exit /b %ERRORLEVEL%
