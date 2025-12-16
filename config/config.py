"""
Configuration file for ISL Hand Gesture Recognition System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATASET_PATH = Path(r"C:\Users\Aakash\Downloads\ISL Hand Gesture Dataset\ISL_Dataset_Extracted\ISL custom Data")

# Model configuration
MODEL_CONFIG = {
    "cnn_backbone": "resnet18",  # Changed to ResNet18 for better accuracy (was mobilenet_v2)
    "input_size": (224, 224),  # Input frame size
    "sequence_length": 16,  # Number of frames per sequence
    "num_classes": None,  # Will be set based on dataset
    "lstm_hidden_size": 256,  # Increased from 128 for better capacity
    "lstm_num_layers": 2,
    "dropout_rate": 0.4,  # Reduced from 0.5 to prevent over-regularization
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,  # Reduced to 8 for CPU training (was 16)
    "epochs": 100,  # Increased from 50 for more training
    "learning_rate": 0.0005,  # Reduced from 0.001 for more stable training
    "weight_decay": 1e-4,
    "patience": 15,  # Increased from 10 to allow more training
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "num_workers": 0,  # Set to 0 for CPU training to avoid multiprocessing issues
    "save_checkpoint_every": 5,  # Save checkpoint every N epochs
    "save_last_checkpoint": True,  # Save last epoch checkpoint
    "warmup_epochs": 5,  # Learning rate warmup epochs
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 10,  # Reduced from 15 to preserve gesture structure
    "horizontal_flip": False,  # Disabled - gestures are directional
    "brightness_range": (0.85, 1.15),  # Reduced range for more realistic augmentation
    "contrast_range": (0.85, 1.15),  # Reduced range
    "zoom_range": (0.95, 1.05),  # Reduced zoom to preserve hand details
    "enable_augmentation": True,
    "gaussian_noise": True,  # Added noise augmentation
    "noise_std": 0.01,  # Standard deviation for noise
}

# Inference configuration
INFERENCE_CONFIG = {
    "webcam_id": 0,
    "frame_skip": 2,  # Process every Nth frame for speed
    "confidence_threshold": 0.5,
    "smoothing_window": 5,  # Number of predictions to average
    "fps": 60,  # Target frame rate
}

# Text-to-Speech configuration
TTS_CONFIG = {
    "language": "hi",  # Hindi (hi), Tamil (ta), Telugu (te), etc.
    "engine": "gtts",  # Options: "gtts", "pyttsx3"
    "slow": False,
}

# Regional language mapping (ISL gesture to regional text)
REGIONAL_MAPPING = {
    # This will be populated based on your dataset labels
    # Example: "A": "अ", "B": "ब", etc.
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

