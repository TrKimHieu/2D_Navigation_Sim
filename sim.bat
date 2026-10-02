@echo off
rem Open a simulation (dataset, map, robot, number of envs, start/goal); watch it or serve it over ZeroMQ.
rem Usage: sim.bat --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\run.ps1" sim hm3denv sim %*
exit /b %ERRORLEVEL%
