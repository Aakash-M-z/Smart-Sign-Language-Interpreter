"""
Evaluation metrics and visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from pathlib import Path
from typing import Dict, List


def calculate_metrics(y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      num_classes: int) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate per-class metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist()
    }
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          label_names: Dict[int, str],
                          save_path: Path = None,
                          figsize: tuple = (12, 10)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Mapping from label index to name
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Get unique labels
    unique_labels = sorted(set(y_true) | set(y_pred))
    label_list = [label_names.get(i, f"Class {i}") for i in unique_labels]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_list, yticklabels=label_list,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list,
                ax=ax2, cbar_kws={'label': 'Normalized'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                label_names: Dict[int, str]):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Mapping from label index to name
    """
    unique_labels = sorted(set(y_true) | set(y_pred))
    target_names = [label_names.get(i, f"Class {i}") for i in unique_labels]
    
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names,
        zero_division=0
    )
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(report)
    print("="*60 + "\n")


def plot_per_class_metrics(metrics: Dict[str, float],
                          label_names: Dict[int, str],
                          save_path: Path = None):
    """
    Plot per-class F1 scores
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        label_names: Mapping from label index to name
        save_path: Path to save the plot
    """
    f1_per_class = metrics['f1_per_class']
    num_classes = len(f1_per_class)
    labels = [label_names.get(i, f"Class {i}") for i in range(num_classes)]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(num_classes), f1_per_class, color='steelblue')
    plt.xlabel('Gesture Class', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Per-Class F1-Scores', fontsize=14, fontweight='bold')
    plt.xticks(range(num_classes), labels, rotation=45, ha='right')
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_per_class)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class metrics plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

