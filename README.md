# SMS Spam Detection - NLP Case Study

A comprehensive Natural Language Processing case study implementing and comparing five different machine learning models for SMS spam classification.

## Project Overview

This project demonstrates the application of NLP techniques for text classification using the SMS Spam Collection dataset. It includes complete preprocessing, model training, evaluation, and comparative analysis.

### Key Features:
- **5 Machine Learning Models**: Logistic Regression, Naive Bayes, Random Forest, SVM, LSTM
- **Complete Preprocessing Pipeline**: Text cleaning, tokenization, stopword removal, lemmatization
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Comparative Analysis**: Visual comparisons and model recommendations

## Dataset

**SMS Spam Collection Dataset**
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) / [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: ~5,574 SMS messages
- **Classes**: Ham (legitimate) ~87%, Spam ~13%
- **Format**: CSV with label and message columns

## Project Structure

```
NLP-CA3/
├── data/
│   ├── raw/                          # Original dataset
│   ├── processed/                    # Preprocessed data
│   └── README.md                     # Dataset documentation
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Data preprocessing steps
│   ├── 03_model_training.ipynb       # Model training and evaluation
│   └── 04_model_comparison.ipynb     # Comparative analysis
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_preprocessing.py         # Text preprocessing functions
│   ├── feature_extraction.py         # TF-IDF and tokenization
│   ├── models.py                     # Model implementations
│   └── evaluation.py                 # Metrics and visualization
├── results/
│   ├── metrics/                      # Performance metrics (JSON, CSV)
│   ├── plots/                        # Visualization outputs
│   └── confusion_matrices/           # Confusion matrix plots
├── models/
│   └── saved_models/                 # Trained model files
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Linux/Mac setup script
├── setup.bat                         # Windows setup script
├── README.md                         # This file
├── TRAINING.md                       # Training guide
├── REPORT.md                         # Case study report
└── LICENSE
```

## Quick Start

### Automated Setup (Recommended)

Run the setup script to automatically configure the environment:

```bash
# Linux/Mac
./setup.sh

# Windows
setup.bat
```

This will:
- Create virtual environment
- Install all dependencies
- Download NLTK data
- Download SMS Spam dataset automatically
- Setup complete - ready to train!

### Manual Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sahilsonii/NLP-CA3.git
cd NLP-CA3
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### 5. Download Dataset

**Automatic (Recommended):**
```bash
python download_dataset.py
```

**Manual:**
Download the SMS Spam Collection dataset and place it in `data/raw/spam.csv`:
- From [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Or use the UCI ML Repository

**📖 For detailed training instructions, see [TRAINING.md](TRAINING.md)**

## Usage

### Run Notebooks in Order:
```bash
jupyter notebook
```

1. **01_data_exploration.ipynb** - Explore and visualize the dataset
2. **02_preprocessing.ipynb** - Clean and preprocess text data
3. **03_model_training.ipynb** - Train and evaluate all 5 models
4. **04_model_comparison.ipynb** - Compare model performances

### Using Source Modules:
```python
from src.data_preprocessing import preprocess_text, load_data
from src.feature_extraction import TFIDFExtractor
from src.models import SpamDetector
from src.evaluation import calculate_metrics, plot_confusion_matrix

# Load and preprocess
df = load_data('data/raw/spam.csv')
df['processed'] = df['message'].apply(preprocess_text)

# Train model
model = SpamDetector('logistic_regression')
model.train(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
metrics = calculate_metrics(y_test, predictions)
```

## Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| Logistic Regression | Traditional ML | Linear classifier using logistic function |
| Naive Bayes | Traditional ML | Probabilistic classifier (MultinomialNB) |
| Random Forest | Ensemble ML | Collection of decision trees |
| SVM | Traditional ML | Support Vector Machine with linear kernel |
| LSTM | Deep Learning | Bidirectional LSTM neural network |

## Results Summary

All models achieved high performance on this dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~97% | ~95% | ~91% | ~93% |
| Naive Bayes | ~96% | ~99% | ~79% | ~88% |
| Random Forest | ~97% | ~100% | ~84% | ~91% |
| SVM | ~98% | ~97% | ~92% | ~94% |
| LSTM | ~98% | ~95% | ~93% | ~94% |

*Note: Exact values may vary based on random seed and environment.*

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Spam detection accuracy (avoid false positives)
- **Recall**: Spam capture rate (avoid false negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

## Key Findings

1. **All models perform well** (>95% accuracy) on this dataset
2. **SVM and LSTM** achieve the best F1-scores
3. **Naive Bayes** is the fastest to train
4. **Traditional ML models** are competitive with deep learning for this task
5. **Preprocessing significantly impacts** model performance

## Technologies Used

- **Python 3.10+**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - Traditional ML models
- **TensorFlow/Keras** - Deep learning (LSTM)
- **NLTK** - Natural language processing
- **Matplotlib, Seaborn** - Visualization
- **Jupyter Notebook** - Interactive development

## Author

**NLP Case Study Project**
- GitHub Repository: [https://github.com/Sahilsonii/NLP-CA3](https://github.com/Sahilsonii/NLP-CA3)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- Scikit-learn and TensorFlow communities
- NLTK project for NLP tools

---

**For the complete case study report, see [REPORT.md](REPORT.md)**