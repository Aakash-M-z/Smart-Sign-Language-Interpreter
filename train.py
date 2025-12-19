"""
Training script for ISL Hand Gesture Recognition System
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Optional

from config.config import (
    MODEL_CONFIG, TRAINING_CONFIG, AUGMENTATION_CONFIG,
    DATASET_PATH, MODELS_DIR, LOGS_DIR
)
from models.cnn_lstm import create_model
from utils.dataset import create_data_loaders
from utils.evaluation import calculate_metrics, plot_confusion_matrix


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """
    Train for one epoch
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate model
    
    Returns:
        Tuple of (average_loss, average_accuracy, all_predictions, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float,
                weight_decay: float,
                patience: int,
                device: torch.device,
                save_dir: Path,
                save_checkpoint_every: int = 5,
                save_last_checkpoint: bool = True,
                resume_from: Optional[Path] = None) -> dict:
    """
    Main training function
    
    Args:
        save_checkpoint_every: Save checkpoint every N epochs (0 to disable)
        save_last_checkpoint: Whether to save last epoch checkpoint
        resume_from: Path to checkpoint to resume from (None to start fresh)
    
    Returns:
        Training history dictionary
    """
    # Loss and optimizer
    # Use standard CrossEntropyLoss (label_smoothing requires PyTorch 1.10+)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))  # Changed to AdamW for better weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )  # Changed to cosine annealing for better convergence
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_from is not None and resume_from.exists():
        print(f"\n{'='*60}")
        print(f"Attempting to resume from checkpoint: {resume_from}")
        print(f"{'='*60}\n")
        
        try:
            checkpoint = torch.load(resume_from, map_location=device)
            
            # Check if checkpoint is compatible with current model architecture
            checkpoint_keys = set(checkpoint['model_state_dict'].keys())
            model_keys = set(model.state_dict().keys())
            
            # Check for key architecture differences
            checkpoint_has_fc3 = 'fc3.weight' in checkpoint_keys
            model_has_fc3 = 'fc3.weight' in model_keys
            checkpoint_has_fc2 = 'fc2.weight' in checkpoint_keys
            
            # If checkpoint has old architecture (no fc3) but model has new architecture (has fc3)
            if not checkpoint_has_fc3 and checkpoint_has_fc2 and model_has_fc3:
                print(f"[INFO] Checkpoint uses old architecture (no fc3 layer)")
                print(f"[INFO] Current model uses new architecture (with fc3, bidirectional LSTM)")
                print(f"[INFO] Attempting partial transfer learning from old checkpoint...")
                print(f"[INFO] Loading compatible parts (CNN backbone) and initializing new layers...\n")
                
                # Try partial loading - load compatible parts
                checkpoint_state = checkpoint['model_state_dict']
                model_state = model.state_dict()
                
                # Load CNN backbone weights (should be compatible)
                loaded_count = 0
                skipped_count = 0
                
                for key in checkpoint_state.keys():
                    try:
                        # Load CNN backbone weights
                        if key.startswith('cnn_backbone.'):
                            if key in model_state:
                                if model_state[key].shape == checkpoint_state[key].shape:
                                    model_state[key] = checkpoint_state[key]
                                    loaded_count += 1
                                else:
                                    skipped_count += 1
                            else:
                                skipped_count += 1
                        # Try to load fc1 if shapes match (might be different due to bidirectional LSTM)
                        elif key == 'fc1.weight' or key == 'fc1.bias':
                            if key in model_state:
                                old_shape = checkpoint_state[key].shape
                                new_shape = model_state[key].shape
                                # If input size matches, we can load (output size might differ)
                                if len(old_shape) == len(new_shape) and old_shape[1] == new_shape[1]:  # Input dimension matches
                                    # Copy what we can
                                    min_out = min(old_shape[0], new_shape[0])
                                    if len(old_shape) == 2:  # Weight matrix
                                        model_state[key][:min_out, :] = checkpoint_state[key][:min_out, :]
                                    else:  # Bias vector
                                        model_state[key][:min_out] = checkpoint_state[key][:min_out]
                                    loaded_count += 1
                                else:
                                    skipped_count += 1
                            else:
                                skipped_count += 1
                    except Exception as e:
                        skipped_count += 1
                        continue
                
                model.load_state_dict(model_state)
                print(f"[OK] Partially loaded checkpoint:")
                print(f"     - Loaded {loaded_count} compatible layers (CNN backbone)")
                print(f"     - Skipped {skipped_count} incompatible layers")
                print(f"     - New layers (fc2, fc3, bidirectional LSTM) initialized from scratch")
                print(f"     - This is transfer learning from old model\n")
                
                # Don't load optimizer state (architecture changed)
                print("[INFO] Starting with fresh optimizer (architecture changed)")
                
                # Use checkpoint epoch info for reference but start from 0
                old_epoch = checkpoint.get('epoch', 0)
                old_best_acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0.0))
                print(f"[INFO] Previous training: Epoch {old_epoch}, Best Val Acc: {old_best_acc:.2f}%")
                print(f"[INFO] Starting fresh training with improved architecture from epoch 0\n")
                
                start_epoch = 0  # Start from beginning
                # Keep history empty to start fresh
                
            else:
                # Try to load the checkpoint normally (same architecture)
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"[OK] Checkpoint loaded successfully")
                    
                    # Load optimizer state if available
                    if 'optimizer_state_dict' in checkpoint:
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        except:
                            print("[WARNING] Could not load optimizer state. Starting with fresh optimizer.")
                    else:
                        print("[WARNING] No optimizer state found in checkpoint. Starting with fresh optimizer.")
                    
                    start_epoch = checkpoint.get('epoch', 0)
                    history = checkpoint.get('history', history)
                    
                    # Restore best values
                    if 'best_val_acc' in checkpoint:
                        history['best_val_acc'] = checkpoint.get('best_val_acc', history.get('best_val_acc', 0.0))
                    if 'best_epoch' in checkpoint:
                        history['best_epoch'] = checkpoint.get('best_epoch', history.get('best_epoch', 0))
                    
                    # Ensure history has best values
                    if 'best_val_acc' not in history:
                        history['best_val_acc'] = checkpoint.get('best_val_acc', 0.0)
                    if 'best_epoch' not in history:
                        history['best_epoch'] = checkpoint.get('best_epoch', 0)
                    
                    print(f"[OK] Resumed from epoch {start_epoch}")
                    print(f"[OK] Best validation accuracy so far: {history['best_val_acc']:.2f}% (epoch {history['best_epoch']})")
                    print(f"[OK] Current learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")
                except Exception as e:
                    print(f"[ERROR] Failed to load checkpoint: {e}")
                    print(f"[INFO] Starting fresh training instead.\n")
                    resume_from = None
                    start_epoch = 0
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint file: {e}")
            print(f"[INFO] Starting fresh training instead.\n")
            resume_from = None
            start_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training on {device}")
    if resume_from:
        print(f"Resuming from epoch {start_epoch + 1}/{num_epochs}")
    else:
        print(f"Training for {num_epochs} epochs")
    print(f"Press Ctrl+C to pause and save checkpoint")
    print(f"{'='*60}\n")
    
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = validate(
                model, val_loader, criterion, device
            )
            
            # Learning rate scheduling (cosine annealing - step by epoch, not by loss)
            scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > history['best_val_acc']:
                history['best_val_acc'] = val_acc
                history['best_epoch'] = epoch + 1
                
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'history': history,
                    'best_val_acc': history['best_val_acc'],
                    'best_epoch': history['best_epoch']
                }
                
                torch.save(checkpoint_data, save_dir / 'best_model.pth')
                print(f"\n[OK] Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Periodic checkpoint saving
            if save_checkpoint_every > 0 and (epoch + 1) % save_checkpoint_every == 0:
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'history': history,
                    'best_val_acc': history['best_val_acc'],
                    'best_epoch': history['best_epoch']
                }
                
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save(checkpoint_data, checkpoint_path)
                print(f"[OK] Saved periodic checkpoint: {checkpoint_path.name}")
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Generate graph and output file for this epoch
            try:
                graph_path = plot_epoch_history(history, epoch + 1, save_dir)
                output_path = save_epoch_output(
                    history, epoch + 1, train_loss, train_acc, 
                    val_loss, val_acc, optimizer.param_groups[0]['lr'], save_dir
                )
                print(f"[OK] Saved epoch {epoch + 1} graph: {graph_path.name}")
                print(f"[OK] Saved epoch {epoch + 1} output: {output_path.name}")
            except Exception as e:
                print(f"Warning: Could not save epoch {epoch + 1} graph/output: {e}")
            
            # Early stopping check (but don't stop too early)
            if epoch + 1 >= 20:  # Don't early stop before 20 epochs
                if early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("Training paused by user (Ctrl+C)")
        print(f"{'='*60}")
        print(f"Saving checkpoint at epoch {epoch + 1}...")
        
        # Save pause checkpoint
        pause_checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc if 'val_acc' in locals() else 0.0,
            'val_loss': val_loss if 'val_loss' in locals() else float('inf'),
            'train_acc': train_acc if 'train_acc' in locals() else 0.0,
            'train_loss': train_loss if 'train_loss' in locals() else float('inf'),
            'history': history,
            'best_val_acc': history['best_val_acc'],
            'best_epoch': history['best_epoch']
        }
        
        pause_checkpoint_path = save_dir / f'paused_checkpoint_epoch_{epoch + 1}.pth'
        torch.save(pause_checkpoint_data, pause_checkpoint_path)
        
        print(f"[OK] Checkpoint saved: {pause_checkpoint_path.name}")
        print(f"\nTo resume training, edit train.py and set:")
        print(f"  resume_from=Path(r\"{pause_checkpoint_path}\")")
        print(f"\nThen run: python train.py")
        print(f"{'='*60}\n")
        
        # Also save as last checkpoint
        torch.save(pause_checkpoint_data, save_dir / 'last_checkpoint.pth')
        print(f"[OK] Also saved as: last_checkpoint.pth\n")
        
        return history
    
    # Save last epoch checkpoint
    if save_last_checkpoint:
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'history': history,
            'best_val_acc': history['best_val_acc'],
            'best_epoch': history['best_epoch']
        }
        
        torch.save(checkpoint_data, save_dir / 'last_checkpoint.pth')
        print(f"\n[OK] Saved last epoch checkpoint: last_checkpoint.pth")
    
    # Load best model
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n[OK] Loaded best model from epoch {checkpoint['epoch']}")
    
    return history


def find_latest_checkpoint(models_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint file across all training directories.
    
    Priority order:
    1. last_checkpoint.pth (most recent epoch)
    2. paused_checkpoint_epoch_N.pth (paused training)
    3. checkpoint_epoch_N.pth (periodic checkpoints)
    4. best_model.pth (best model, but may not be the latest epoch)
    
    Returns:
        Path to the latest checkpoint, or None if no checkpoint found
    """
    if not models_dir.exists():
        return None
    
    latest_checkpoint = None
    latest_epoch = -1
    latest_time = 0
    
    # Search all training directories
    for training_dir in models_dir.glob("training_*"):
        if not training_dir.is_dir():
            continue
        
        # Check for last_checkpoint.pth (highest priority)
        last_checkpoint = training_dir / "last_checkpoint.pth"
        if last_checkpoint.exists():
            try:
                checkpoint = torch.load(last_checkpoint, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                mtime = last_checkpoint.stat().st_mtime
                if epoch > latest_epoch or (epoch == latest_epoch and mtime > latest_time):
                    latest_checkpoint = last_checkpoint
                    latest_epoch = epoch
                    latest_time = mtime
            except Exception as e:
                print(f"Warning: Could not load {last_checkpoint}: {e}")
        
        # Check for paused checkpoints
        for paused_checkpoint in training_dir.glob("paused_checkpoint_epoch_*.pth"):
            try:
                checkpoint = torch.load(paused_checkpoint, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                mtime = paused_checkpoint.stat().st_mtime
                if epoch > latest_epoch or (epoch == latest_epoch and mtime > latest_time):
                    latest_checkpoint = paused_checkpoint
                    latest_epoch = epoch
                    latest_time = mtime
            except Exception as e:
                print(f"Warning: Could not load {paused_checkpoint}: {e}")
        
        # Check for periodic checkpoints
        for periodic_checkpoint in training_dir.glob("checkpoint_epoch_*.pth"):
            try:
                checkpoint = torch.load(periodic_checkpoint, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                mtime = periodic_checkpoint.stat().st_mtime
                if epoch > latest_epoch or (epoch == latest_epoch and mtime > latest_time):
                    latest_checkpoint = periodic_checkpoint
                    latest_epoch = epoch
                    latest_time = mtime
            except Exception as e:
                print(f"Warning: Could not load {periodic_checkpoint}: {e}")
        
        # Check for best_model.pth as fallback (lowest priority)
        best_model = training_dir / "best_model.pth"
        if best_model.exists():
            try:
                checkpoint = torch.load(best_model, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                mtime = best_model.stat().st_mtime
                # Only use best_model if no other checkpoint found or if it's newer
                if latest_checkpoint is None or (epoch > latest_epoch or (epoch == latest_epoch and mtime > latest_time)):
                    latest_checkpoint = best_model
                    latest_epoch = epoch
                    latest_time = mtime
            except Exception as e:
                print(f"Warning: Could not load {best_model}: {e}")
    
    return latest_checkpoint


def plot_training_history(history: dict, save_path: Path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved training history plot to {save_path / 'training_history.png'}")


def plot_epoch_history(history: dict, epoch: int, save_path: Path):
    """Plot training history up to current epoch and save"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title(f'Model Loss - Epoch {epoch}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_title(f'Model Accuracy - Epoch {epoch}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    epoch_graph_path = save_path / f'epoch_{epoch}_history.png'
    plt.savefig(epoch_graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    return epoch_graph_path


def save_epoch_output(history: dict, epoch: int, train_loss: float, train_acc: float,
                     val_loss: float, val_acc: float, learning_rate: float, save_path: Path):
    """Save epoch output to JSON file"""
    epoch_data = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'train_acc': float(train_acc),
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
        'learning_rate': float(learning_rate),
        'best_val_acc': float(history.get('best_val_acc', 0.0)),
        'best_epoch': history.get('best_epoch', 0),
        'timestamp': datetime.now().isoformat()
    }
    
    epoch_output_path = save_path / f'epoch_{epoch}_output.json'
    with open(epoch_output_path, 'w') as f:
        json.dump(epoch_data, f, indent=2)
    
    return epoch_output_path


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint(MODELS_DIR)
    
    # Determine save directory
    if latest_checkpoint is not None:
        # Resume in the same training directory
        save_dir = latest_checkpoint.parent
        print(f"\n{'='*60}")
        print(f"Found latest checkpoint: {latest_checkpoint.name}")
        print(f"Resuming training in: {save_dir.name}")
        print(f"{'='*60}\n")
    else:
        # Create new training directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = MODELS_DIR / f"training_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nNo checkpoint found. Starting new training in: {save_dir.name}\n")
    
    # Create data loaders
    print("\nLoading dataset...")
    try:
        train_loader, val_loader, test_loader, label_mapping = create_data_loaders(
            data_path=DATASET_PATH,
            sequence_length=MODEL_CONFIG["sequence_length"],
            input_size=MODEL_CONFIG["input_size"],
            batch_size=TRAINING_CONFIG["batch_size"],
            train_split=TRAINING_CONFIG["train_split"],
            val_split=TRAINING_CONFIG["val_split"],
            test_split=TRAINING_CONFIG["test_split"],
            augmentation_config=AUGMENTATION_CONFIG,
            num_workers=TRAINING_CONFIG["num_workers"]
        )
    except ValueError as e:
        print(f"\n{'='*60}")
        print("[ERROR] Dataset Loading Failed")
        print(f"{'='*60}")
        print(str(e))
        print(f"\nPlease check:")
        print(f"  1. Dataset path in config/config.py: {DATASET_PATH}")
        print(f"  2. Dataset structure contains gesture folders with video/image files")
        print(f"  3. File permissions allow reading the dataset")
        print(f"\nExpected structure:")
        print(f"  dataset_root/")
        print(f"    ├── Gesture1/")
        print(f"    │   ├── video1.mp4  OR  image1.jpg, image2.jpg, ...")
        print(f"    │   └── ...")
        print(f"    ├── Gesture2/")
        print(f"    └── ...")
        print(f"{'='*60}\n")
        return
    
    # Update model config with number of classes
    MODEL_CONFIG["num_classes"] = label_mapping["num_classes"]
    
    # Create model
    print(f"\nCreating model with {label_mapping['num_classes']} classes...")
    model = create_model(MODEL_CONFIG, label_mapping["num_classes"])
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAINING_CONFIG["epochs"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        patience=TRAINING_CONFIG["patience"],
        device=device,
        save_dir=save_dir,
        save_checkpoint_every=TRAINING_CONFIG.get("save_checkpoint_every", 5),
        save_last_checkpoint=TRAINING_CONFIG.get("save_last_checkpoint", True),
        resume_from=latest_checkpoint  # Automatically resume from latest checkpoint
    )
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}\n")
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    # Calculate metrics
    metrics = calculate_metrics(test_labels, test_preds, label_mapping["num_classes"])
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_labels, 
        test_preds, 
        label_mapping["idx_to_label"],
        save_path=save_dir / 'confusion_matrix.png'
    )
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    # Save configuration and results
    config_save = {
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'label_mapping': label_mapping,
        'test_metrics': {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro'])
        },
        'training_history': {
            'best_val_acc': float(history['best_val_acc']),
            'best_epoch': history['best_epoch'],
            'total_epochs': len(history['train_loss'])
        }
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    from typing import Tuple
    main()

