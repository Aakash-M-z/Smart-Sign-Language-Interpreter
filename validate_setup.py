"""
Validation script to check if the project setup is correct and ready for training
"""
import sys
from pathlib import Path
import importlib

def check_imports():
    """Check if all required packages are installed"""
    print("="*70)
    print("Checking Required Packages")
    print("="*70)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }
    
    optional = {
        'gtts': 'gTTS (for TTS)',
        'pygame': 'Pygame (for audio)',
        'pyttsx3': 'pyttsx3 (alternative TTS)'
    }
    
    all_ok = True
    
    print("\nRequired packages:")
    for module, name in required.items():
        try:
            importlib.import_module(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} - MISSING")
            all_ok = False
    
    print("\nOptional packages:")
    for module, name in optional.items():
        try:
            importlib.import_module(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [WARN] {name} - Not installed (optional)")
    
    return all_ok


def check_project_structure():
    """Check if project structure is correct"""
    print("\n" + "="*70)
    print("Checking Project Structure")
    print("="*70)
    
    project_root = Path(__file__).parent
    required_files = [
        'config/config.py',
        'models/cnn_lstm.py',
        'utils/dataset.py',
        'utils/preprocessing.py',
        'utils/evaluation.py',
        'utils/tts_engine.py',
        'train.py',
        'realtime_inference.py',
        'evaluate_model.py',
        'requirements.txt'
    ]
    
    all_ok = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - MISSING")
            all_ok = False
    
    return all_ok


def check_config():
    """Check if configuration is valid"""
    print("\n" + "="*70)
    print("Checking Configuration")
    print("="*70)
    
    try:
        from config.config import (
            MODEL_CONFIG, TRAINING_CONFIG, AUGMENTATION_CONFIG,
            DATASET_PATH, MODELS_DIR
        )
        
        print(f"  [OK] Config loaded successfully")
        print(f"  [OK] CNN Backbone: {MODEL_CONFIG['cnn_backbone']}")
        print(f"  [OK] LSTM Hidden Size: {MODEL_CONFIG['lstm_hidden_size']}")
        print(f"  [OK] Batch Size: {TRAINING_CONFIG['batch_size']}")
        print(f"  [OK] Learning Rate: {TRAINING_CONFIG['learning_rate']}")
        print(f"  [OK] Dataset Path: {DATASET_PATH}")
        print(f"  [OK] Dataset Exists: {DATASET_PATH.exists()}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error loading config: {e}")
        return False


def check_model_architecture():
    """Check if model can be created"""
    print("\n" + "="*70)
    print("Checking Model Architecture")
    print("="*70)
    
    try:
        from config.config import MODEL_CONFIG
        from models.cnn_lstm import create_model
        
        # Create a dummy model
        num_classes = 26  # A-Z
        model = create_model(MODEL_CONFIG, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  [OK] Model created successfully")
        print(f"  [OK] Total parameters: {total_params:,}")
        print(f"  [OK] Trainable parameters: {trainable_params:,}")
        print(f"  [OK] Model has bidirectional LSTM: {model.lstm.bidirectional}")
        print(f"  [OK] Model has batch normalization: {hasattr(model, 'bn1')}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset():
    """Check if dataset is accessible"""
    print("\n" + "="*70)
    print("Checking Dataset")
    print("="*70)
    
    try:
        from config.config import DATASET_PATH
        
        if not DATASET_PATH.exists():
            print(f"  [FAIL] Dataset path does not exist: {DATASET_PATH}")
            print(f"     Please run: python setup_dataset.py")
            return False
        
        print(f"  [OK] Dataset path exists: {DATASET_PATH}")
        
        # Check for gesture folders
        gesture_folders = [d for d in DATASET_PATH.iterdir() if d.is_dir()]
        
        if len(gesture_folders) == 0:
            print(f"  [WARN] No gesture folders found")
            print(f"     Expected structure: dataset_root/GestureName/videos_or_images")
            return False
        
        print(f"  [OK] Found {len(gesture_folders)} gesture folders")
        
        # Check for files in first few folders
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        image_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        total_files = 0
        for folder in gesture_folders[:5]:
            folder_files = sum(
                len(list(folder.rglob(f"*{ext}"))) for ext in video_exts + image_exts
            )
            total_files += folder_files
            if folder_files > 0:
                print(f"    [OK] {folder.name}: {folder_files} files")
            else:
                print(f"    [WARN] {folder.name}: No files found")
        
        if total_files == 0:
            print(f"  [WARN] No video/image files found in dataset")
            return False
        
        print(f"  [OK] Dataset appears to be valid")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Error checking dataset: {e}")
        return False


def main():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("ISL Gesture Recognition - Setup Validation")
    print("="*70)
    
    checks = [
        ("Package Imports", check_imports),
        ("Project Structure", check_project_structure),
        ("Configuration", check_config),
        ("Model Architecture", check_model_architecture),
        ("Dataset", check_dataset),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n  ✗ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("[SUCCESS] All checks passed! Project is ready for training.")
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Monitor training progress")
        print("  3. Evaluate: python evaluate_model.py")
        return 0
    else:
        print("[FAIL] Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Extract dataset: python setup_dataset.py")
        print("  - Check dataset path in config/config.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())

