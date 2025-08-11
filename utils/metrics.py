"""
Evaluation metrics for HGAT-LDA model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_score: Prediction scores (continuous)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['auc'] = roc_auc_score(y_true, y_score)
    metrics['aupr'] = average_precision_score(y_true, y_score)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    metrics['precision'] = precision
    metrics['recall'] = recall
    
    # F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    metrics['f1_max'] = np.max(f1_scores)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Additional metrics using sklearn functions for robustness
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Handle edge cases where only one class is present
    if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
        metrics['sensitivity'] = 0.0
        metrics['specificity'] = 0.0
        metrics['precision_score'] = 0.0
        metrics['recall_score'] = 0.0
        metrics['f1_score'] = 0.0
    else:
        # Calculate metrics using sklearn functions
        metrics['precision_score'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall_score'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate sensitivity and specificity from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            metrics['sensitivity'] = 0.0
            metrics['specificity'] = 0.0
    
    return metrics


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC score
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, aupr: float, save_path: Optional[str] = None):
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        aupr: AUPR score
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results(metrics: Dict[str, float], results_path: str):
    """
    Save evaluation results to file.
    
    Args:
        metrics: Dictionary of metrics
        results_path: Path to save results
    """
    # Convert numpy arrays to lists for JSON serialization
    results = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
        else:
            results[key] = value
    
    # Save as JSON
    import json
    with open(f"{results_path}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV for tabular metrics
    tabular_metrics = {k: v for k, v in results.items() 
                      if not isinstance(v, list) and not k in ['confusion_matrix']}
    df = pd.DataFrame([tabular_metrics])
    df.to_csv(f"{results_path}.csv", index=False)
    
    print(f"Results saved to {results_path}.json and {results_path}.csv")


def print_metrics_summary(metrics: Dict[str, float]):
    """
    Print a summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    
    print(f"AUC:                    {metrics['auc']:.4f}")
    print(f"AUPR:                   {metrics['aupr']:.4f}")
    print(f"F1 Score (Max):         {metrics['f1_max']:.4f}")
    print(f"Sensitivity:            {metrics['sensitivity']:.4f}")
    print(f"Specificity:            {metrics['specificity']:.4f}")
    print(f"Precision:              {metrics['precision_score']:.4f}")
    print(f"Recall:                 {metrics['recall_score']:.4f}")
    print(f"F1 Score:               {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*50)


def calculate_loocv_statistics(auc_scores: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for LOOCV results.
    
    Args:
        auc_scores: List of AUC scores from LOOCV
        
    Returns:
        Dictionary of statistics
    """
    auc_scores = np.array(auc_scores)
    
    stats = {
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores),
        'min_auc': np.min(auc_scores),
        'max_auc': np.max(auc_scores),
        'median_auc': np.median(auc_scores),
        'num_folds': len(auc_scores)
    }
    
    return stats


def plot_loocv_results(auc_scores: List[float], save_path: Optional[str] = None):
    """
    Plot LOOCV results.
    
    Args:
        auc_scores: List of AUC scores from LOOCV
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(auc_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(auc_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(auc_scores):.3f}')
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.title('LOOCV AUC Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(auc_scores)
    plt.ylabel('AUC Score')
    plt.title('LOOCV AUC Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show() 