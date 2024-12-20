@echo off

REM Download and install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | more"

REM Create virtual environment using uv
uv venv derma_env

REM Activate virtual environment
call derma_env\Scripts\activate.bat

REM Install requirements using uv pip
uv pip install -r requirements.txt

REM No need for deactivate in batch file
