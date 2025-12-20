"""
Quick start script to test the ISL Gesture Recognition System setup
"""
import sys
from pathlib import Path
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }
    
    optional_packages = {
        'gtts': 'gTTS (for text-to-speech)',
        'pygame': 'Pygame (for audio playback)',
        'pyttsx3': 'pyttsx3 (alternative TTS engine)'
    }
    
    print("="*60)
    print("Checking Dependencies")
    print("="*60)
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    print("\nRequired packages:")
    for module, name in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing_required.append(name)
    
    # Check optional packages
    print("\nOptional packages:")
    for module, name in optional_packages.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING (optional)")
            missing_optional.append(name)
    
    print("\n" + "="*60)
    
    if missing_required:
        print(f"\nERROR: Missing required packages: {', '.join(missing_required)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        if missing_optional:
            print(f"\nNote: Optional packages missing: {', '.join(missing_optional)}")
            print("These are needed for TTS functionality.")
        return True


def check_project_structure():
    """Check if project structure is correct"""
    print("\n" + "="*60)
    print("Checking Project Structure")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    required_dirs = ['config', 'models', 'utils', 'data', 'logs']
    required_files = [
        'config/config.py',
        'models/cnn_lstm.py',
        'utils/dataset.py',
        'utils/preprocessing.py',
        'utils/evaluation.py',
        'utils/tts_engine.py',
        'train.py',
        'realtime_inference.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    print("\nDirectories:")
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            all_good = False
    
    print("\nFiles:")
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} - MISSING")
            all_good = False
    
    print("\n" + "="*60)
    
    if all_good:
        print("\n✓ Project structure is correct!")
    else:
        print("\n✗ Some files/directories are missing!")
    
    return all_good


def check_dataset():
    """Check if dataset is available"""
    print("\n" + "="*60)
    print("Checking Dataset")
    print("="*60)
    
    from config.config import DATASET_PATH
    
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"\n✗ Dataset path does not exist: {dataset_path}")
        print("  Please update DATASET_PATH in config/config.py")
        return False
    
    print(f"\nDataset path: {dataset_path}")
    
    # Check for gesture folders
    gesture_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not gesture_folders:
        print("  ✗ No gesture folders found in dataset directory")
        return False
    
    print(f"  ✓ Found {len(gesture_folders)} gesture folders")
    
    # Check for files in folders
    total_files = 0
    for folder in gesture_folders[:5]:  # Check first 5 folders
        files = list(folder.glob("*"))
        total_files += len(files)
        print(f"    - {folder.name}: {len(files)} files")
    
    if len(gesture_folders) > 5:
        print(f"    ... and {len(gesture_folders) - 5} more folders")
    
    print("\n" + "="*60)
    print("\n✓ Dataset is available!")
    
    return True


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("ISL Gesture Recognition System - Quick Start Check")
    print("="*60)
    
    deps_ok = check_dependencies()
    structure_ok = check_project_structure()
    dataset_ok = check_dataset()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if deps_ok and structure_ok and dataset_ok:
        print("\n✓ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Train the model: python train.py")
        print("  2. Run real-time inference: python realtime_inference.py --model <model_path>")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        
        if not deps_ok:
            print("\n  → Install dependencies: pip install -r requirements.txt")
        if not structure_ok:
            print("\n  → Ensure all project files are present")
        if not dataset_ok:
            print("\n  → Update DATASET_PATH in config/config.py")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

