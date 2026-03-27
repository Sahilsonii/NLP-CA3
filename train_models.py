#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMS Spam Detection - Model Training Script
Trains all 5 models and evaluates their performance
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import load_data, preprocess_pipeline
from feature_extraction import TFIDFExtractor, SequenceExtractor
from models import SpamDetector
from evaluation import calculate_metrics, print_metrics as print_eval_metrics

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def parse_args():
    """Parse command-line options for training verbosity."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate SMS spam detection models."
    )
    parser.add_argument(
        "--sklearn-verbose",
        type=int,
        default=1,
        help="Native verbosity level for supported sklearn models. Use 0 to disable.",
    )
    parser.add_argument(
        "--lstm-verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Keras verbosity for LSTM training: 0=silent, 1=progress bar, 2=one line per epoch.",
    )
    return parser.parse_args()


def get_metric(metrics, key):
    """Return a metric value with backward-compatible aliases."""
    aliases = {
        "f1_score": ("f1_score", "f1"),
        "f1": ("f1", "f1_score"),
    }

    for candidate in aliases.get(key, (key,)):
        if candidate in metrics:
            return metrics[candidate]

    raise KeyError(key)


def print_model_results(model_name, metrics, training_time):
    """Print model performance metrics"""
    f1_value = get_metric(metrics, "f1_score")
    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {f1_value:.4f} ({f1_value*100:.2f}%)")
    if 'roc_auc' in metrics:
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   ⏱️  Training time: {training_time:.2f} seconds")


def print_training_diagnostics(model_name, diagnostics):
    """Print structured training details."""
    if not diagnostics:
        return

    print(f"   Training details for {model_name}:")
    for key, value in diagnostics.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"   - {label}: {value:.4f}")
        else:
            print(f"   - {label}: {value}")


def main():
    """Main training pipeline"""
    args = parse_args()

    print_header("🚀 SMS Spam Detection - Model Training Pipeline")

    # Step 1: Load data
    print("\n[1/6] 📥 Loading data...")
    try:
        df = load_data('data/raw/spam.csv')
        print(f"   ✓ Loaded {len(df)} messages")
        print(f"   ✓ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # Step 2: Preprocess data
    print("\n[2/6] 🔧 Preprocessing text...")
    try:
        # The dataset has 'v1' (label) and 'v2' (message)
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        df = df[['label', 'message']]  # Keep only these columns

        df = preprocess_pipeline(df, text_column='message')
        print(f"   ✓ Preprocessed {len(df)} messages")

        # Convert labels to binary
        df['label'] = (df['label'] == 'spam').astype(int)
        print(f"   ✓ Spam messages: {df['label'].sum()}")
        print(f"   ✓ Ham messages: {(df['label'] == 0).sum()}")
    except Exception as e:
        print(f"   ❌ Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Split data
    print("\n[3/6] ✂️  Splitting data...")
    try:
        X = df['processed_text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"   ✓ Training samples: {len(X_train)}")
        print(f"   ✓ Test samples: {len(X_test)}")
    except Exception as e:
        print(f"   ❌ Error splitting data: {e}")
        return

    # Step 4: Extract features
    print("\n[4/6] 🎯 Extracting features...")
    try:
        # TF-IDF for traditional ML
        print("   🔄 Extracting TF-IDF features...")
        tfidf = TFIDFExtractor(max_features=5000)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        print(f"   ✓ TF-IDF features: {X_train_tfidf.shape[1]} dimensions")

        # Sequences for LSTM
        print("   🔄 Extracting sequence features...")
        seq_extractor = SequenceExtractor(max_words=10000, max_length=100)
        X_train_seq = seq_extractor.fit_transform(X_train)
        X_test_seq = seq_extractor.transform(X_test)
        print(f"   ✓ Sequence features: max length {X_train_seq.shape[1]}")
        vocab_size = seq_extractor.get_vocabulary_size()
        print(f"   ✓ Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"   ❌ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Train models
    print_header("[5/6] 🎯 Training Models")
    print("Traditional ML models do not train in epochs, so the script reports solver/tree diagnostics for them.")
    print("The LSTM will print one summary line per epoch so you can track training and validation metrics clearly.\n")
    if args.sklearn_verbose > 0:
        print(f"Native sklearn verbose logging is enabled at level {args.sklearn_verbose}.")
    else:
        print("Native sklearn verbose logging is disabled.")
    print(f"LSTM verbose mode is set to {args.lstm_verbose}.\n")

    models_config = [
        (
            "logistic_regression",
            "Logistic Regression",
            X_train_tfidf,
            X_test_tfidf,
            {"sklearn_verbose": args.sklearn_verbose},
        ),
        (
            "naive_bayes",
            "Multinomial Naive Bayes",
            X_train_tfidf,
            X_test_tfidf,
            {"sklearn_verbose": args.sklearn_verbose},
        ),
        (
            "random_forest",
            "Random Forest",
            X_train_tfidf,
            X_test_tfidf,
            {"sklearn_verbose": args.sklearn_verbose},
        ),
        (
            "svm",
            "Support Vector Machine",
            X_train_tfidf,
            X_test_tfidf,
            {"sklearn_verbose": args.sklearn_verbose},
        ),
    ]

    results = {}
    trained_models = {}

    # Train traditional ML models
    for model_type, model_name, X_tr, X_te, kwargs in models_config:
        print(f"\n🔄 Training {model_name}...")
        try:
            detector = SpamDetector(model_type=model_type)
            detector.train(X_tr, y_train.values, **kwargs)
            print_training_diagnostics(model_name, detector.get_training_diagnostics())

            # Predict
            y_pred = detector.predict(X_te)
            y_proba = detector.predict_proba(X_te)

            # Evaluate
            metrics = calculate_metrics(y_test.values, y_pred, y_proba[:, 1])
            results[model_name] = metrics
            trained_models[model_name] = detector

            print_model_results(model_name, metrics, detector.training_time)

        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Train LSTM model
    print("\n" + "="*70)
    print("🧠 Training LSTM Neural Network")
    print("="*70)
    print("\n⚠️  Note: LSTM training may take 5-10 minutes on CPU...\n")

    try:
        detector = SpamDetector(model_type="lstm")

        print(f"🔄 Training LSTM (vocab size: {vocab_size})...")
        detector.train(
            X_train_seq, y_train.values,
            vocab_size=vocab_size,
            max_length=X_train_seq.shape[1],
            epochs=10,
            batch_size=32,
            verbose=args.lstm_verbose,
            use_class_weight=False,
        )
        print_training_diagnostics("LSTM Neural Network", detector.get_training_diagnostics())

        # Predict
        y_pred = detector.predict(X_test_seq)
        y_proba = detector.predict_proba(X_test_seq)

        # Evaluate
        metrics = calculate_metrics(y_test.values, y_pred, y_proba[:, 1])
        results["LSTM Neural Network"] = metrics
        trained_models["LSTM Neural Network"] = detector

        print_model_results("LSTM Neural Network", metrics, detector.training_time)

    except Exception as e:
        print(f"   ❌ Error training LSTM: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Save models
    print_header("[6/6] 💾 Saving Trained Models")

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)

    save_mapping = {
        "Logistic Regression": "logistic_regression.pkl",
        "Multinomial Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "Support Vector Machine": "svm.pkl",
        "LSTM Neural Network": "lstm_model.keras"
    }

    for model_name, filename in save_mapping.items():
        if model_name in trained_models:
            try:
                filepath = models_dir / filename
                trained_models[model_name].save(str(filepath))
                print(f"   ✓ Saved {model_name} to {filepath}")
            except Exception as e:
                print(f"   ❌ Error saving {model_name}: {e}")

    # Save feature extractors
    try:
        import joblib
        joblib.dump(tfidf, str(models_dir / 'tfidf_vectorizer.pkl'))
        joblib.dump(seq_extractor, str(models_dir / 'sequence_tokenizer.pkl'))
        print(f"   ✓ Saved feature extractors")
    except Exception as e:
        print(f"   ❌ Error saving extractors: {e}")

    if "LSTM Neural Network" in trained_models:
        detector = trained_models["LSTM Neural Network"]
        if detector.history is not None:
            history_dir = Path("results/metrics")
            history_dir.mkdir(parents=True, exist_ok=True)
            history_path = history_dir / "lstm_training_history.csv"
            pd.DataFrame(detector.history.history).to_csv(history_path, index_label="epoch")
            print(f"   ✓ Saved LSTM training history to {history_path}")

    # Final summary
    print_header("📊 Training Complete - Model Comparison")

    if results:
        print("\n" + "=" * 90)
        print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("=" * 90)

        for model_name, metrics in results.items():
            f1_value = get_metric(metrics, "f1_score")
            print(f"{model_name:<30} "
                  f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:>5.2f}%)  "
                  f"{metrics['precision']:.4f} ({metrics['precision']*100:>5.2f}%)  "
                  f"{metrics['recall']:.4f} ({metrics['recall']*100:>5.2f}%)  "
                  f"{f1_value:.4f} ({f1_value*100:>5.2f}%)")

        print("=" * 90)

        # Find best model
        best_model = max(results.items(), key=lambda x: get_metric(x[1], "f1_score"))
        print(f"\n🏆 Best Model: {best_model[0]}")
        best_f1 = get_metric(best_model[1], "f1_score")
        print(f"   F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")

    print("\n✅ Training pipeline completed successfully!")
    print("\n📁 Saved files:")
    print("   - models/*.pkl (Traditional ML models)")
    print("   - models/lstm_model.keras (Deep learning model)")
    print("   - models/tfidf_vectorizer.pkl (TF-IDF vectorizer)")
    print("   - models/sequence_tokenizer.pkl (Sequence tokenizer)")
    print("\n📊 Next steps:")
    print("   - Run Jupyter notebooks for detailed EDA and visualizations")
    print("   - Check notebooks/04_model_comparison.ipynb for in-depth analysis")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
