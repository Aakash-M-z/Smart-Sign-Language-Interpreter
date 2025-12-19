"""
Standalone script to evaluate a trained model on the test set
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json

from config.config import (
    MODEL_CONFIG, TRAINING_CONFIG, DATASET_PATH, MODELS_DIR
)
from models.cnn_lstm import create_model
from utils.dataset import create_data_loaders
from utils.evaluation import calculate_metrics, plot_confusion_matrix, plot_per_class_metrics


def evaluate_model(model_path: Path = None, verbose: int = 1):
    """
    Evaluate a trained model on the test set
    
    Args:
        model_path: Path to the model checkpoint. If None, uses the latest best_model.pth
        verbose: Verbosity level (0 or 1)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose >= 1:
        print(f"Using device: {device}")
    
    # Find model path
    if model_path is None:
        # Find the latest best_model.pth
        training_dirs = sorted(MODELS_DIR.glob("training_*"), reverse=True)
        for training_dir in training_dirs:
            best_model_path = training_dir / "best_model.pth"
            if best_model_path.exists():
                model_path = best_model_path
                if verbose >= 1:
                    print(f"Found model: {model_path}")
                break
        
        if model_path is None:
            raise FileNotFoundError("No trained model found. Please train a model first or specify model_path.")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create data loaders
    if verbose >= 1:
        print("\nLoading dataset...")
    train_loader, val_loader, test_loader, label_mapping = create_data_loaders(
        data_path=DATASET_PATH,
        sequence_length=MODEL_CONFIG["sequence_length"],
        input_size=MODEL_CONFIG["input_size"],
        batch_size=TRAINING_CONFIG["batch_size"],
        train_split=TRAINING_CONFIG["train_split"],
        val_split=TRAINING_CONFIG["val_split"],
        test_split=TRAINING_CONFIG["test_split"],
        augmentation_config={},  # No augmentation for evaluation
        num_workers=TRAINING_CONFIG["num_workers"]
    )
    
    # Update model config with number of classes
    MODEL_CONFIG["num_classes"] = label_mapping["num_classes"]
    
    # Create model
    model = create_model(MODEL_CONFIG, label_mapping["num_classes"])
    model = model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if verbose >= 1:
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Model has {label_mapping['num_classes']} classes")
        print(f"\nEvaluating on test set...")
    
    # Evaluate on test set
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate additional metrics
    metrics = calculate_metrics(
        all_labels, 
        all_preds, 
        label_mapping["num_classes"]
    )
    
    # Print results
    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}% | 📉 Test Loss: {test_loss:.4f}")
    
    if verbose >= 1:
        print(f"\nDetailed Metrics:")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        
        # Save directory
        save_dir = model_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Generating visualization graphs...")
        print(f"{'='*60}")
        
        # 1. Confusion Matrix
        plot_confusion_matrix(
            all_labels,
            all_preds,
            label_mapping["idx_to_label"],
            save_path=save_dir / 'test_confusion_matrix.png'
        )
        
        # 2. Per-class F1 Scores
        plot_per_class_metrics(
            metrics,
            label_mapping["idx_to_label"],
            save_path=save_dir / 'test_per_class_f1.png'
        )
        
        # 3. Per-class Precision, Recall, F1 Comparison
        plot_per_class_comparison(
            all_labels,
            all_preds,
            label_mapping["idx_to_label"],
            save_path=save_dir / 'test_per_class_metrics.png'
        )
        
        # 4. Overall Metrics Summary
        plot_metrics_summary(
            metrics,
            test_accuracy,
            test_loss,
            save_path=save_dir / 'test_metrics_summary.png'
        )
        
        # 5. Per-class Accuracy
        plot_per_class_accuracy(
            all_labels,
            all_preds,
            label_mapping["idx_to_label"],
            save_path=save_dir / 'test_per_class_accuracy.png'
        )
        
        # 6. Training History (if available in checkpoint)
        if 'history' in checkpoint:
            plot_training_history_from_checkpoint(
                checkpoint['history'],
                save_path=save_dir / 'training_history.png'
            )
        
        # 7. Save metrics to JSON
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'metrics': {
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'precision_macro': float(metrics['precision_macro']),
                'precision_weighted': float(metrics['precision_weighted']),
                'recall_macro': float(metrics['recall_macro']),
                'recall_weighted': float(metrics['recall_weighted']),
                'f1_per_class': [float(x) for x in metrics['f1_per_class']]
            },
            'model_info': {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'num_classes': label_mapping["num_classes"],
                'model_path': str(model_path)
            }
        }
        
        with open(save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("✓ All graphs and results saved successfully!")
        print(f"  Saved to: {save_dir}")
        print(f"{'='*60}\n")
    
    return test_loss, test_accuracy, metrics


def plot_per_class_comparison(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              label_names: dict,
                              save_path: Path = None):
    """Plot per-class precision, recall, and F1 scores comparison"""
    num_classes = len(label_names)
    labels = [label_names.get(i, f"Class {i}") for i in range(num_classes)]
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Gesture Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics Comparison (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only show label if significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class metrics comparison to {save_path.name}")
    plt.close()


def plot_metrics_summary(metrics: dict,
                         accuracy: float,
                         loss: float,
                         save_path: Path = None):
    """Plot overall metrics summary"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall Metrics Bar Chart
    metric_names = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision (Macro)', 'Recall (Macro)']
    metric_values = [
        accuracy / 100,  # Convert to 0-1 scale
        metrics['f1_macro'],
        metrics['f1_weighted'],
        metrics['precision_macro'],
        metrics['recall_macro']
    ]
    
    bars = ax1.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'], alpha=0.8)
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Metrics Summary', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Loss Display
    ax2.text(0.5, 0.5, f'{loss:.4f}', 
            ha='center', va='center', fontsize=48, fontweight='bold', color='#e74c3c')
    ax2.set_title('Test Loss', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Accuracy Display
    ax3.text(0.5, 0.5, f'{accuracy:.2f}%', 
            ha='center', va='center', fontsize=48, fontweight='bold', color='#2ecc71')
    ax3.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. F1 Scores Distribution
    f1_per_class = metrics['f1_per_class']
    ax4.hist(f1_per_class, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(f1_per_class), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_per_class):.3f}')
    ax4.axvline(np.median(f1_per_class), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(f1_per_class):.3f}')
    ax4.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('F1-Score Distribution Across Classes', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Evaluation Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics summary to {save_path.name}")
    plt.close()


def plot_per_class_accuracy(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            label_names: dict,
                            save_path: Path = None):
    """Plot per-class accuracy"""
    num_classes = len(label_names)
    labels = [label_names.get(i, f"Class {i}") for i in range(num_classes)]
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(num_classes), per_class_accuracy, color='steelblue', alpha=0.8)
    ax.set_xlabel('Gesture Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.3f}',
               ha='center', va='bottom', fontsize=9)
    
    # Add average line
    avg_acc = np.mean(per_class_accuracy)
    ax.axhline(avg_acc, color='red', linestyle='--', linewidth=2, 
              label=f'Average: {avg_acc:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class accuracy to {save_path.name}")
    plt.close()


def plot_training_history_from_checkpoint(history: dict,
                                         save_path: Path = None):
    """Plot training history from checkpoint"""
    if not history or 'train_loss' not in history:
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path.name}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to model checkpoint (default: latest best_model.pth)"
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        default=1,
        help="Verbosity level (0 or 1, default: 1)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model) if args.model else None
    test_loss, test_accuracy, metrics = evaluate_model(model_path, args.verbose)

