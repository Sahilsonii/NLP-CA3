# SMS Spam Detection - NLP Case Study

A comprehensive Natural Language Processing project implementing and comparing **5 different machine learning models** for SMS spam classification.

## Overview

This project demonstrates text classification using the SMS Spam Collection dataset, featuring complete preprocessing pipelines, model training, evaluation, and comparative analysis.

### Models Implemented
- **Logistic Regression** - Linear classifier
- **Naive Bayes** - Probabilistic classifier
- **Random Forest** - Ensemble learning
- **SVM** - Support Vector Machine
- **LSTM** - Deep learning with Bidirectional LSTM

### Key Features
- Complete text preprocessing pipeline (cleaning, tokenization, lemmatization)
- TF-IDF and sequential feature extraction
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Interactive Jupyter notebooks for exploration
- Automated setup scripts

## Quick Start

### 1. Automated Setup (Recommended)

Run the setup script to automatically configure everything:

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Download NLTK data
- Download the SMS Spam dataset automatically

**Total setup time:** 5-10 minutes

### 2. Manual Setup

If you prefer manual installation:

```bash
# Clone the repository
git clone https://github.com/Sahilsonii/NLP-CA3.git
cd NLP-CA3

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_dataset.py
```

### 3. Verify Installation

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/spam.csv', encoding='latin-1'); print(f'✓ Dataset loaded: {len(df)} rows')"
```

Expected output: `✓ Dataset loaded: 5572 rows`

## Usage

### Option 1: Train All Models (Command Line)

```bash
# Activate environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Train all models
python train_models.py
```

This will:
- Load and preprocess the dataset
- Train all 5 models
- Save models to `models/`
- Generate evaluation metrics in `results/`
- Display performance comparison

### Option 2: Interactive Notebooks

```bash
# Activate environment and launch Jupyter
source venv/bin/activate
jupyter notebook
```

Run notebooks in order:
1. `01_data_exploration.ipynb` - Explore the dataset
2. `02_preprocessing.ipynb` - Text preprocessing
3. `03_model_training.ipynb` - Train models
4. `04_model_comparison.ipynb` - Compare results

### Option 3: Use as Python Module

```python
from src.data_preprocessing import preprocess_text, load_data
from src.feature_extraction import TFIDFExtractor
from src.models import SpamDetector
from src.evaluation import calculate_metrics

# Load and preprocess data
df = load_data('data/raw/spam.csv')
df['processed'] = df['message'].apply(preprocess_text)

# Train a model
model = SpamDetector('logistic_regression')
model.train(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
metrics = calculate_metrics(y_test, predictions)
```

## Project Structure

```
NLP-CA3/
├── data/
│   ├── raw/                    # Original dataset (gitignored)
│   └── processed/              # Preprocessed data (gitignored)
├── notebooks/                  # Jupyter notebooks (4 notebooks)
├── src/
│   ├── data_preprocessing.py   # Text cleaning & preprocessing
│   ├── feature_extraction.py   # TF-IDF & tokenization
│   ├── models.py               # Model implementations
│   └── evaluation.py           # Metrics & visualization
├── models/                     # Trained models (gitignored)
├── results/                    # Outputs & metrics (gitignored)
├── train_models.py             # Main training script
├── download_dataset.py         # Dataset download utility
├── requirements.txt            # Python dependencies
├── setup.sh / setup.bat        # Setup scripts
├── README.md                   # This file
├── TRAINING.md                 # Detailed training guide
└── REPORT.md                   # Case study report
```

## Results

All models achieve excellent performance (>95% accuracy):

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | ~97% | ~95% | ~91% | ~93% | < 1s |
| Naive Bayes | ~96% | ~99% | ~79% | ~88% | < 1s |
| Random Forest | ~97% | ~100% | ~84% | ~91% | 2-5s |
| SVM | ~98% | ~97% | ~92% | ~94% | 10-30s |
| LSTM | ~98% | ~95% | ~93% | ~94% | 5-10min |

**Recommendations:**
- **Best F1-Score:** SVM, LSTM
- **Fastest:** Logistic Regression, Naive Bayes
- **Best Precision:** Random Forest (minimizes false positives)

## Dataset

**SMS Spam Collection Dataset**
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size:** 5,574 SMS messages
- **Distribution:** Ham (87%), Spam (13%)
- **Format:** CSV with label and message columns

The dataset is automatically downloaded by the setup script or can be manually downloaded using `python download_dataset.py`.

## Technologies

- **Python 3.10+**
- **Data Processing:** Pandas, NumPy
- **ML Models:** Scikit-learn
- **Deep Learning:** TensorFlow/Keras
- **NLP:** NLTK
- **Visualization:** Matplotlib, Seaborn

## Troubleshooting

### Dataset not found
```bash
python download_dataset.py
```

### Import errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Windows: venv\Scripts\activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### NLTK data missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### Model training errors
See [TRAINING.md](TRAINING.md) for detailed troubleshooting and advanced options.

## Documentation

- **README.md** (this file) - Quick start and overview
- **TRAINING.md** - Detailed training guide with advanced options
- **REPORT.md** - Complete case study report with analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Sahil Soni**
- GitHub: [@Sahilsonii](https://github.com/Sahilsonii)
- Repository: [NLP-CA3](https://github.com/Sahilsonii/NLP-CA3)

## Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- Scikit-learn and TensorFlow communities
- NLTK project for NLP tools

---

**Ready to start?** Run `./setup.sh` (or `setup.bat` on Windows) and then `python train_models.py`!
