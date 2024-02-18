@echo off

call sd_scripts\venv\Scripts\activate
call uvicorn main:app
pause