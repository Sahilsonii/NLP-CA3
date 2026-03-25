"""
Evaluation Module for SMS Spam Detection

This module provides functions for model evaluation:
- Metrics calculation (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix plotting
- ROC curve plotting
- Model comparison visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import os


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all evaluation metrics.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1:
            proba = y_pred_proba[:, 1]
        else:
            proba = y_pred_proba
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted way."""
    print(f"
{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    print(f"{'='*50}")


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix as heatmap.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path=None):
    """
    Plot ROC curve.
    
    Parameters:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the figure (optional)
    """
    if len(y_pred_proba.shape) > 1:
        proba = y_pred_proba[:, 1]
    else:
        proba = y_pred_proba
    
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    plt.close()


def compare_models(results_dict, save_path=None):
    """
    Create comparison bar chart for all models.
    
    Parameters:
        results_dict: Dictionary with model names as keys and metrics dict as values
        save_path: Path to save the figure (optional)
    """
    df = pd.DataFrame(results_dict).T
    
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(df))
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        bars = ax.bar(df.index, df[metric], color=colors)
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        
        ax.tick_params(axis="x", rotation=45)
    
    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved to {save_path}")
    
    plt.show()
    plt.close()


def create_comparison_table(results_dict):
    """
    Create a formatted comparison table.
    
    Parameters:
        results_dict: Dictionary with model names as keys and metrics dict as values
        
    Returns:
        pd.DataFrame: Comparison table
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    
    if "training_time" in df.columns:
        df["training_time"] = df["training_time"].apply(lambda x: f"{x:.2f}s")
    
    df.index.name = "Model"
    return df


def plot_all_roc_curves(results_dict, y_true, save_path=None):
    """
    Plot ROC curves for all models on the same figure.
    
    Parameters:
        results_dict: Dictionary with model predictions
        y_true: True labels
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(results_dict))
    
    for (model_name, data), color in zip(results_dict.items(), colors):
        if "y_pred_proba" in data:
            proba = data["y_pred_proba"]
            if len(proba.shape) > 1:
                proba = proba[:, 1]
            fpr, tpr, _ = roc_curve(y_true, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")
    
    plt.show()
    plt.close()


def save_results_to_csv(results_dict, filepath):
    """Save results to CSV file."""
    df = create_comparison_table(results_dict)
    df.to_csv(filepath)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100, 2)
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, "Demo Model")
