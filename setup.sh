#!/bin/bash

# SMS Spam Detection - Quick Setup Script
# This script automates the setup process for the NLP-CA3 project

set -e  # Exit on error

echo "=========================================="
echo "SMS Spam Detection - Environment Setup"
echo "=========================================="
echo

# Step 1: Check Python installation
echo "[1/5] Checking Python installation..."
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Python found: $PYTHON_VERSION"
echo

# Step 2: Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
fi
echo

# Step 3: Activate virtual environment
echo "[3/5] Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"
echo

# Step 4: Upgrade pip and install dependencies
echo "[4/5] Installing dependencies..."
echo "This may take several minutes (TensorFlow is large)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo

# Step 5: Download NLTK data
echo "[5/5] Downloading NLTK data..."
python -c "
import nltk
import sys

packages = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
for package in packages:
    try:
        nltk.download(package, quiet=True)
        print(f'✓ Downloaded: {package}')
    except Exception as e:
        print(f'✗ Failed to download {package}: {e}', file=sys.stderr)
"
echo

# Final instructions
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Download the SMS Spam dataset:"
echo "   https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset"
echo
echo "2. Place spam.csv in: data/raw/spam.csv"
echo
echo "3. Start training:"
echo "   - For notebooks: jupyter notebook"
echo "   - For scripts: python src/models.py"
echo
echo "For detailed instructions, see TRAINING.md"
echo
echo "Virtual environment is activated."
echo "To deactivate, run: deactivate"
echo
