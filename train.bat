@echo off
rem Train a PPO agent (Stable-Baselines3) and evaluate it on unseen maps.
rem Usage: train.bat --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\win\run.ps1" train examples\train_ppo.py %*
exit /b %ERRORLEVEL%
