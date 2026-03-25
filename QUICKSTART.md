# Quick Start Guide

## 🚀 Fully Automated Setup

The project now includes **complete automation** for environment setup and dataset download!

### One-Command Setup

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
./setup.sh
```

This single command will:
1. ✅ Check Python installation
2. ✅ Create virtual environment (`venv/`)
3. ✅ Activate virtual environment
4. ✅ Install all dependencies (pandas, scikit-learn, tensorflow, nltk, etc.)
5. ✅ Download NLTK data (stopwords, punkt, wordnet, etc.)
6. ✅ **Automatically download SMS Spam dataset** (5,572 messages)
7. ✅ Verify dataset integrity

**Total time:** 5-10 minutes (depending on internet speed)

## 📥 Dataset Download

The dataset (spam.csv) is now automatically downloaded!

### Automatic Download Script

```bash
python download_dataset.py
```

**Features:**
- Tries Kaggle API first (if credentials configured)
- Falls back to alternative GitHub mirror
- Verifies dataset (5,572 rows, ~492 KB)
- Handles errors gracefully with helpful messages

**Dataset Info:**
- **Total messages:** 5,572
- **Ham (legitimate):** 4,825 (86.6%)
- **Spam:** 747 (13.4%)
- **Location:** `data/raw/spam.csv`

### Optional: Kaggle API Setup

For faster downloads in future:

1. Create account at https://www.kaggle.com
2. Go to Settings → API → Create New Token
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
4. Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

## 🎯 Start Training

After setup completes, you're immediately ready to train!

### Option 1: Jupyter Notebooks (Recommended for Learning)

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate  # Windows: venv\Scripts\activate

# Launch Jupyter
jupyter notebook
```

Run notebooks in order:
1. `01_exploratory_data_analysis.ipynb` - Explore the dataset
2. `02_data_preprocessing.ipynb` - Preprocess the text
3. `03_model_training.ipynb` - Train all 5 models
4. `04_model_comparison.ipynb` - Compare performances

### Option 2: Python Scripts (Production)

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run training scripts
cd src
python models.py
```

## 📊 Expected Results

All 5 models achieve >95% accuracy:

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Logistic Regression | ~97% | < 1 second |
| Naive Bayes | ~96% | < 1 second |
| Random Forest | ~97% | 2-5 seconds |
| SVM | ~98% | 10-30 seconds |
| LSTM | ~98% | 5-10 minutes |

## 📚 Documentation

- **TRAINING.md** - Comprehensive training guide with troubleshooting
- **README.md** - Project overview and quick start
- **REPORT.md** - Full case study report with analysis

## 🎉 What's New

### Latest Updates (v2.0)

1. **Automated Dataset Download**
   - No more manual downloads!
   - `download_dataset.py` script with Kaggle API support
   - Automatic fallback to GitHub mirror
   - Dataset verification built-in

2. **Enhanced Setup Scripts**
   - `setup.sh` and `setup.bat` now fully automated
   - Includes dataset download as step 6/6
   - Better error handling and progress messages

3. **Complete Documentation**
   - Added detailed dataset download instructions
   - Kaggle API setup guide
   - Multiple fallback options documented

4. **Dataset Included**
   - SMS Spam Collection dataset ready to use
   - Pre-verified (5,572 messages)
   - No download step required for git users

## 🔧 Troubleshooting

### Dataset Issues

**Problem:** Dataset not found after setup
**Solution:**
```bash
python download_dataset.py
```

**Problem:** Kaggle API not working
**Solution:** Script automatically uses alternative download source

### Environment Issues

**Problem:** Virtual environment not activated
**Solution:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Problem:** Package installation failed
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🌟 Success Criteria

You're ready to train when:
- ✅ Virtual environment is activated
- ✅ All dependencies installed (check: `pip list`)
- ✅ NLTK data downloaded
- ✅ `data/raw/spam.csv` exists (check: `ls data/raw/`)
- ✅ Dataset loads without errors

**Verify everything:**
```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/spam.csv', encoding='latin-1'); print(f'✅ Dataset loaded: {len(df)} rows')"
```

Expected output: `✅ Dataset loaded: 5572 rows`

## 🚀 Ready to Go!

Everything is set up and ready. Start training with:

```bash
jupyter notebook
```

Happy training! 🎓

---

**Repository:** https://github.com/Sahilsonii/NLP-CA3
**Latest Commit:** Add automatic dataset download functionality
