@echo off
REM SMS Spam Detection - Quick Setup Script (Windows)
REM This script automates the setup process for the NLP-CA3 project

echo ==========================================
echo SMS Spam Detection - Environment Setup
echo ==========================================
echo.

REM Step 1: Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python 3.10 or higher.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [32mv Python found: %PYTHON_VERSION%[0m
echo.

REM Step 2: Create virtual environment
echo [2/5] Creating virtual environment...
if exist venv\ (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo [32mv Virtual environment created[0m
)
echo.

REM Step 3: Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo [32mv Virtual environment activated[0m
echo.

REM Step 4: Upgrade pip and install dependencies
echo [4/5] Installing dependencies...
echo This may take several minutes (TensorFlow is large)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo [32mv Dependencies installed[0m
echo.

REM Step 5: Download NLTK data
echo [5/6] Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True); print('v NLTK data downloaded')"
echo.

REM Step 6: Download SMS Spam dataset
echo [6/6] Downloading SMS Spam dataset...
python download_dataset.py
echo.

REM Final instructions
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Everything is ready! You can now start training:
echo.
echo Option 1 - Jupyter Notebooks (Recommended for learning):
echo   jupyter notebook
echo   Then run notebooks in order: 01, 02, 03, 04
echo.
echo Option 2 - Python Scripts (For production):
echo   cd src
echo   python models.py
echo.
echo For detailed instructions, see TRAINING.md
echo.
echo Virtual environment is activated.
echo To deactivate, run: deactivate
echo.
pause
