# Model Training Guide

This guide provides step-by-step instructions for training the SMS Spam Detection models in a virtual environment.

## Prerequisites

- Python 3.10 or higher
- Git
- At least 4GB of free disk space
- Internet connection for downloading dependencies and dataset

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sahilsonii/NLP-CA3.git
cd NLP-CA3
```

### 2. Create Virtual Environment

Create a virtual environment to isolate project dependencies:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages:
- pandas, numpy (Data manipulation)
- scikit-learn (ML algorithms)
- tensorflow (Deep learning for LSTM)
- nltk (NLP preprocessing)
- matplotlib, seaborn, wordcloud (Visualization)
- jupyter, notebook (Interactive notebooks)

### 4. Download NLTK Data

Download required NLTK datasets:

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Download the SMS Spam Dataset

**Automated Download (Recommended)**

Use the provided download script:

```bash
python download_dataset.py
```

This script will:
- Try to download from Kaggle API (if credentials are configured)
- Fall back to alternative download source if needed
- Automatically place the file in `data/raw/spam.csv`
- Verify the dataset integrity

**Manual Download Options**

If automatic download doesn't work:

**Option 1: Kaggle (Recommended)**

1. Visit: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download `spam.csv`
3. Place it in `data/raw/spam.csv`

**Option 2: UCI Repository**

1. Visit: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
2. Download the dataset
3. Extract and rename to `spam.csv`
4. Place it in `data/raw/spam.csv`

**Kaggle API Setup (Optional)**

To use Kaggle API for faster downloads:

1. Create Kaggle account at https://www.kaggle.com/
2. Go to Account Settings → API
3. Click "Create New Token" → downloads `kaggle.json`
4. Place `kaggle.json` in:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
5. Linux/Mac only: `chmod 600 ~/.kaggle/kaggle.json`
6. Run: `python download_dataset.py`

### 6. Verify Dataset

Check if the dataset is loaded correctly:

```python
python -c "import pandas as pd; df = pd.read_csv('data/raw/spam.csv', encoding='latin-1'); print(f'Dataset shape: {df.shape}'); print(df.head())"
```

Expected output: Dataset with 5,574 rows

## Training the Models

### Option 1: Using Python Scripts (Recommended for Production)

Train all models using the modular source code:

```bash
# Navigate to src directory
cd src

# Run the complete training pipeline
python -c "
from data_preprocessing import load_and_preprocess_data
from feature_extraction import extract_features
from models import train_all_models
from evaluation import evaluate_models

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('../data/raw/spam.csv')

# Extract features
X_train_tfidf, X_test_tfidf, X_train_seq, X_test_seq = extract_features(X_train, X_test)

# Train models
models = train_all_models(X_train_tfidf, y_train)

# Evaluate models
results = evaluate_models(models, X_test_tfidf, y_test)
print(results)
"
```

Or run individual model training:

```python
# Train Logistic Regression
python -c "
from data_preprocessing import load_and_preprocess_data
from feature_extraction import extract_features
from models import train_logistic_regression
from evaluation import evaluate_model

X_train, X_test, y_train, y_test = load_and_preprocess_data('../data/raw/spam.csv')
X_train_tfidf, X_test_tfidf, _, _ = extract_features(X_train, X_test)

model = train_logistic_regression(X_train_tfidf, y_train)
results = evaluate_model(model, X_test_tfidf, y_test, 'Logistic Regression')
print(results)
"
```

### Option 2: Using Jupyter Notebooks (Recommended for Learning)

Launch Jupyter and run the notebooks in order:

```bash
jupyter notebook
```

Open and run notebooks in this sequence:

1. **01_exploratory_data_analysis.ipynb**
   - Data loading and inspection
   - Class distribution analysis
   - Message length analysis
   - Word frequency analysis
   - Word clouds visualization

2. **02_data_preprocessing.ipynb**
   - Text cleaning
   - Tokenization
   - Stop word removal
   - Lemmatization
   - Train-test split

3. **03_model_training.ipynb**
   - Feature extraction (TF-IDF, Tokenization)
   - Training 5 models:
     - Logistic Regression
     - Multinomial Naive Bayes
     - Random Forest
     - Support Vector Machine (SVM)
     - LSTM Neural Network
   - Model evaluation

4. **04_model_comparison.ipynb**
   - Performance comparison
   - Confusion matrices
   - ROC curves
   - Feature importance analysis

## Training Time Estimates

Approximate training times on a standard laptop (Intel i5, 8GB RAM):

| Model | Training Time |
|-------|---------------|
| Logistic Regression | < 1 second |
| Naive Bayes | < 1 second |
| Random Forest | 2-5 seconds |
| SVM | 10-30 seconds |
| LSTM | 5-10 minutes |

## Model Output

Trained models are saved in the `models/` directory:

```
models/
├── logistic_regression.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── svm.pkl
├── lstm_model.keras
└── tokenizer.pkl
```

## Expected Results

After training, you should see results similar to:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 96-97% | 95-96% | 85-88% | 90-92% |
| Naive Bayes | 96-97% | 94-96% | 82-86% | 88-91% |
| Random Forest | 97-98% | 96-98% | 85-89% | 90-93% |
| SVM | 97-98% | 96-97% | 88-91% | 92-94% |
| LSTM | 98-99% | 97-98% | 92-95% | 94-96% |

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: NLTK Data Not Found

**Solution:** Download NLTK data:
```python
python -c "import nltk; nltk.download('all')"
```

### Issue: Memory Error During LSTM Training

**Solution:** Reduce batch size or max sequence length in the LSTM training code:
```python
# In src/models.py or notebook
batch_size = 16  # Reduce from 32
max_len = 100    # Reduce from 200
```

### Issue: Dataset Not Found

**Solution:** Verify the file path:
```bash
ls data/raw/spam.csv
```

If missing, re-download following step 5.

### Issue: TensorFlow Warnings

**Solution:** TensorFlow may show warnings about CPU instructions. These can be safely ignored or suppressed:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

## Hyperparameter Tuning

To tune hyperparameters for better performance:

### Logistic Regression

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### Random Forest

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
```

### LSTM

```python
# Adjust architecture in src/models.py
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),  # Increase embedding dim
    LSTM(128, return_sequences=True),  # Add another LSTM layer
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## Re-training Models

To retrain models with updated data:

1. Update the dataset in `data/raw/spam.csv`
2. Delete old models from `models/` directory
3. Run the training pipeline again
4. Compare results with previous models

## Deployment

Once training is complete, you can:

1. **Save Models**: Models are automatically saved in `models/` directory
2. **Load Models**: Use `joblib.load()` for scikit-learn models or `tf.keras.models.load_model()` for LSTM
3. **Make Predictions**:
   ```python
   import joblib

   model = joblib.load('models/logistic_regression.pkl')
   message = ["Congratulations! You've won a prize"]
   prediction = model.predict(message)
   ```

## Performance Monitoring

Track model performance over time:

```python
import json
from datetime import datetime

# Save training metrics
metrics = {
    'timestamp': datetime.now().isoformat(),
    'accuracy': 0.97,
    'precision': 0.96,
    'recall': 0.88,
    'f1_score': 0.92
}

with open('models/training_history.json', 'a') as f:
    json.dump(metrics, f)
    f.write('\n')
```

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [NLTK Documentation](https://www.nltk.org/)
- [SMS Spam Collection Paper](https://dl.acm.org/doi/10.1145/2133806.2133815)

## Contributing

To contribute improvements:

1. Create a new branch for your changes
2. Train models with your modifications
3. Document changes and performance impact
4. Submit a pull request with results

## License

This project is for educational purposes. The dataset is publicly available from UCI Machine Learning Repository.

---

**Last Updated**: March 2026
**Author**: Sahil Soni
**Repository**: https://github.com/Sahilsonii/NLP-CA3
