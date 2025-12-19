"""
Dataset exploration script to find actual dataset structure
"""
from pathlib import Path
import os

def explore_directory(path: Path, max_depth: int = 5, current_depth: int = 0, found_files: list = None):
    """
    Recursively explore directory structure and find all files
    
    Args:
        path: Path to explore
        max_depth: Maximum depth to explore
        current_depth: Current depth level
        found_files: List to store found files
    """
    if found_files is None:
        found_files = []
    
    if current_depth > max_depth:
        return found_files
    
    if not path.exists():
        print(f"  Path does not exist: {path}")
        return found_files
    
    if not path.is_dir():
        return found_files
    
    try:
        items = list(path.iterdir())
        
        # Check for files in current directory
        files_in_dir = [item for item in items if item.is_file()]
        if files_in_dir:
            print(f"\n{'  ' * current_depth}[DIR] {path.name}/ (contains {len(files_in_dir)} files)")
            for file in files_in_dir[:10]:  # Show first 10 files
                print(f"  {'  ' * (current_depth + 1)}[FILE] {file.name}")
                found_files.append(file)
            if len(files_in_dir) > 10:
                print(f"  {'  ' * (current_depth + 1)}... and {len(files_in_dir) - 10} more files")
        else:
            # Only show directory if it has subdirectories
            subdirs = [item for item in items if item.is_dir()]
            if subdirs:
                print(f"\n{'  ' * current_depth}[DIR] {path.name}/ ({len(subdirs)} subdirectories)")
        
        # Recursively explore subdirectories
        for item in items:
            if item.is_dir():
                explore_directory(item, max_depth, current_depth + 1, found_files)
            elif item.is_file():
                if item not in found_files:
                    found_files.append(item)
    
    except PermissionError:
        print(f"  [ERROR] Permission denied: {path}")
    except Exception as e:
        print(f"  [ERROR] Error exploring {path}: {e}")
    
    return found_files


def analyze_dataset_structure(dataset_path: Path):
    """
    Analyze the dataset structure and provide recommendations
    """
    print("="*70)
    print("Dataset Structure Explorer")
    print("="*70)
    print(f"\nExploring: {dataset_path}")
    print(f"Path exists: {dataset_path.exists()}")
    print(f"Path is directory: {dataset_path.is_dir() if dataset_path.exists() else 'N/A'}")
    print("\n" + "="*70)
    print("Directory Structure:")
    print("="*70)
    
    found_files = explore_directory(dataset_path, max_depth=5)
    
    # Analyze found files
    print("\n" + "="*70)
    print("File Analysis:")
    print("="*70)
    
    if not found_files:
        print("\n[ERROR] No files found in the dataset directory!")
        print("\nPossible reasons:")
        print("  1. Dataset is not extracted yet")
        print("  2. Dataset is in a different location")
        print("  3. Files are in a nested structure that needs exploration")
        return
    
    # Group files by extension
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    videos = [f for f in found_files if f.suffix in video_extensions]
    images = [f for f in found_files if f.suffix in image_extensions]
    other_files = [f for f in found_files if f.suffix not in video_extensions | image_extensions]
    
    print(f"\n[SUMMARY]")
    print(f"  Total files found: {len(found_files)}")
    print(f"  Video files: {len(videos)}")
    print(f"  Image files: {len(images)}")
    print(f"  Other files: {len(other_files)}")
    
    if videos:
        print(f"\n[VIDEOS] Video files by folder:")
        video_folders = {}
        for video in videos:
            folder = video.parent.name
            if folder not in video_folders:
                video_folders[folder] = []
            video_folders[folder].append(video)
        
        for folder, files in sorted(video_folders.items()):
            print(f"  {folder}: {len(files)} videos")
    
    if images:
        print(f"\n[IMAGES] Image files by folder:")
        image_folders = {}
        for image in images:
            folder = image.parent.name
            if folder not in image_folders:
                image_folders[folder] = []
            image_folders[folder].append(image)
        
        for folder, files in sorted(image_folders.items()):
            print(f"  {folder}: {len(files)} images")
    
    # Find gesture folders (folders with videos or images)
    print("\n" + "="*70)
    print("Gesture Folders Detected:")
    print("="*70)
    
    gesture_folders = set()
    for file in found_files:
        if file.suffix in video_extensions | image_extensions:
            gesture_folders.add(file.parent)
    
    if gesture_folders:
        print(f"\n[OK] Found {len(gesture_folders)} gesture folders:")
        for folder in sorted(gesture_folders):
            folder_files = [f for f in found_files if f.parent == folder]
            videos_count = len([f for f in folder_files if f.suffix in video_extensions])
            images_count = len([f for f in folder_files if f.suffix in image_extensions])
            print(f"  [FOLDER] {folder.name}: {videos_count} videos, {images_count} images")
            print(f"     Path: {folder}")
    
    # Recommendation
    print("\n" + "="*70)
    print("Recommendations:")
    print("="*70)
    
    if gesture_folders:
        # Find common parent
        common_parents = set()
        for folder in gesture_folders:
            common_parents.add(folder.parent)
        
        if len(common_parents) == 1:
            recommended_path = list(common_parents)[0]
            print(f"\n[OK] Recommended DATASET_PATH:")
            print(f"   {recommended_path}")
            print(f"\n   Update config/config.py with:")
            print(f"   DATASET_PATH = Path(r\"{recommended_path}\")")
        else:
            print(f"\n[WARNING] Gesture folders are in different parent directories.")
            print(f"   Consider organizing them under a single parent directory.")
    else:
        print(f"\n[ERROR] No gesture folders detected.")
        print(f"   Please ensure your dataset contains:")
        print(f"   - Folders named after gestures (e.g., 'A', 'B', 'Hello')")
        print(f"   - Each folder contains video files (.mp4, .avi) or image files (.jpg, .png)")


def main():
    """Main function"""
    from config.config import DATASET_PATH
    
    print("\n" + "="*70)
    print("ISL Dataset Structure Explorer")
    print("="*70)
    
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset path does not exist: {dataset_path}")
        print(f"\nPlease check config/config.py and update DATASET_PATH")
        return
    
    analyze_dataset_structure(dataset_path)
    
    print("\n" + "="*70)
    print("Exploration Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

