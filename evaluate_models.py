"""
Comprehensive Model Evaluation Script
Generates confusion matrices, evaluation metrics, and benchmark comparison
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Ensure output directories exist
os.makedirs('results/confusion_matrices', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)


def load_and_prepare_data():
    """Load data and prepare test set."""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    # Load raw data
    df = pd.read_csv('data/raw/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_encoded'] = (df['label'] == 'spam').astype(int)

    # Add preprocessing imports here after verifying they exist
    import sys
    sys.path.insert(0, 'src')
    from data_preprocessing import preprocess_text

    print("Preprocessing text data...")
    df['processed'] = df['message'].apply(preprocess_text)

    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label_encoded'],
        test_size=0.2, random_state=42, stratify=df['label_encoded']
    )

    print(f"Test set size: {len(X_test)} samples")
    print(f"  - Ham: {(y_test == 0).sum()}")
    print(f"  - Spam: {(y_test == 1).sum()}")

    return X_test, y_test


def load_models_and_vectorizers():
    """Load all trained models and vectorizers."""
    print("\n" + "=" * 70)
    print("LOADING TRAINED MODELS")
    print("=" * 70)

    models = {}

    # Load TF-IDF extractor (custom wrapper class)
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    print("[OK] TF-IDF extractor loaded")

    # Load sequence extractor (custom wrapper class)
    seq_extractor = joblib.load('models/sequence_tokenizer.pkl')
    print("[OK] Sequence extractor loaded")

    # Load sklearn models
    sklearn_models = ['logistic_regression', 'naive_bayes', 'random_forest', 'svm']
    for model_name in sklearn_models:
        filepath = f'models/{model_name}.pkl'
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
            print(f"[OK] {model_name.replace('_', ' ').title()} loaded")

    # Load LSTM model
    try:
        from tensorflow.keras.models import load_model
        models['lstm'] = load_model('models/lstm_model.keras')
        print("[OK] LSTM model loaded")
    except Exception as e:
        print(f"[!] Could not load LSTM model: {e}")

    return models, tfidf, seq_extractor


def evaluate_model(model, X_test, y_test, model_name, is_lstm=False, tokenizer=None):
    """Evaluate a single model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)

    if is_lstm:
        y_pred = (y_pred > 0.5).astype(int).flatten()
        y_proba = model.predict(X_test).flatten()
    else:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test) if hasattr(model, 'decision_function') else None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)

    return y_pred, y_proba, metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham (0)', 'Spam (1)'],
                yticklabels=['Ham (0)', 'Spam (1)'],
                annot_kws={'size': 16})

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_confusion_matrices(all_results, y_test, save_path):
    """Plot all confusion matrices in a single figure."""
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, data['y_pred'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    annot_kws={'size': 14})

        ax.set_title(model_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # Hide extra subplot
    if n_models < 6:
        for i in range(n_models, 6):
            axes[i].set_visible(False)

    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(all_results, y_test, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))

    for (model_name, data), color in zip(all_results.items(), colors):
        if data['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, data['y_proba'])
            roc_auc = auc(fpr, tpr)
            label = f"{model_name.replace('_', ' ').title()} (AUC = {roc_auc:.4f})"
            plt.plot(fpr, tpr, color=color, lw=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(all_metrics, save_path):
    """Plot bar chart comparing all metrics."""
    df = pd.DataFrame(all_metrics).T
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available = [m for m in metrics if m in df.columns]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    for i, metric in enumerate(available):
        values = df[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in df.index], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_benchmark_table(all_metrics):
    """Create a comprehensive benchmark table."""
    df = pd.DataFrame(all_metrics).T

    # Format columns
    for col in df.columns:
        if col != 'training_time':
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

    df.index.name = 'Model'
    df.index = df.index.str.replace('_', ' ').str.title()

    return df


def print_detailed_classification_report(y_test, y_pred, model_name):
    """Print detailed classification report."""
    print(f"\n--- Classification Report: {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("   SMS SPAM DETECTION - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    # Load data
    X_test, y_test = load_and_prepare_data()

    # Load models
    models, tfidf, seq_extractor = load_models_and_vectorizers()

    # Prepare features
    X_test_tfidf = tfidf.transform(X_test)

    # Sequence features for LSTM - use the SequenceExtractor's transform method
    X_test_seq = seq_extractor.transform(X_test)

    # Evaluate all models
    print("\n" + "=" * 70)
    print("EVALUATING MODELS")
    print("=" * 70)

    all_results = {}
    all_metrics = {}

    model_configs = {
        'logistic_regression': ('Logistic Regression', X_test_tfidf, False),
        'naive_bayes': ('Naive Bayes', X_test_tfidf, False),
        'random_forest': ('Random Forest', X_test_tfidf, False),
        'svm': ('SVM', X_test_tfidf, False),
        'lstm': ('LSTM', X_test_seq, True),
    }

    for model_key, (model_name, X_data, is_lstm) in model_configs.items():
        if model_key not in models:
            continue

        print(f"\nEvaluating {model_name}...")
        model = models[model_key]

        y_pred, y_proba, metrics = evaluate_model(
            model, X_data, y_test.values, model_name, is_lstm
        )

        all_results[model_key] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': metrics
        }
        all_metrics[model_key] = metrics

        # Print metrics
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Plot individual confusion matrix
        plot_confusion_matrix(
            y_test.values, y_pred, model_name,
            f'results/confusion_matrices/{model_key}_cm.png'
        )
        print(f"  -> Confusion matrix saved")

    # Generate comparison plots
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # All confusion matrices
    plot_all_confusion_matrices(all_results, y_test.values, 'results/plots/all_confusion_matrices.png')
    print("[OK] All confusion matrices saved to results/plots/all_confusion_matrices.png")

    # ROC curves
    plot_roc_curves(all_results, y_test.values, 'results/plots/roc_curves.png')
    print("[OK] ROC curves saved to results/plots/roc_curves.png")

    # Metrics comparison
    plot_metrics_comparison(all_metrics, 'results/plots/metrics_comparison.png')
    print("[OK] Metrics comparison saved to results/plots/metrics_comparison.png")

    # Create and print benchmark table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    benchmark_df = create_benchmark_table(all_metrics)
    print("\n" + benchmark_df.to_string())

    # Save results
    with open('results/metrics/all_results.json', 'w') as f:
        # Convert numpy to python types for JSON
        json_metrics = {}
        for k, v in all_metrics.items():
            json_metrics[k] = {mk: float(mv) for mk, mv in v.items()}
        json.dump(json_metrics, f, indent=2)
    print("\n[OK] Results saved to results/metrics/all_results.json")

    # Save benchmark CSV
    benchmark_df.to_csv('results/metrics/benchmark_comparison.csv')
    print("[OK] Benchmark saved to results/metrics/benchmark_comparison.csv")

    # Print classification reports
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 70)

    for model_key, data in all_results.items():
        model_name = model_key.replace('_', ' ').title()
        print_detailed_classification_report(y_test.values, data['y_pred'], model_name)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 70)

    # Find best model for each metric
    best_accuracy = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    best_precision = max(all_metrics.items(), key=lambda x: x[1]['precision'])
    best_recall = max(all_metrics.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])

    print(f"\n[BEST] Best Accuracy:  {best_accuracy[0].replace('_', ' ').title()} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"[BEST] Best Precision: {best_precision[0].replace('_', ' ').title()} ({best_precision[1]['precision']:.4f})")
    print(f"[BEST] Best Recall:    {best_recall[0].replace('_', ' ').title()} ({best_recall[1]['recall']:.4f})")
    print(f"[BEST] Best F1-Score:  {best_f1[0].replace('_', ' ').title()} ({best_f1[1]['f1_score']:.4f})")

    if all(('roc_auc' in m) for m in all_metrics.values()):
        best_auc = max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
        print(f"[BEST] Best ROC-AUC:   {best_auc[0].replace('_', ' ').title()} ({best_auc[1]['roc_auc']:.4f})")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  * results/confusion_matrices/ - Individual confusion matrices")
    print("  * results/plots/all_confusion_matrices.png")
    print("  * results/plots/roc_curves.png")
    print("  * results/plots/metrics_comparison.png")
    print("  * results/metrics/all_results.json")
    print("  * results/metrics/benchmark_comparison.csv")


if __name__ == "__main__":
    main()
