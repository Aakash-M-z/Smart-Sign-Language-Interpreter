"""
Dataset diagnostic script to explore dataset structure and find data files
"""
from pathlib import Path
import os

def explore_directory(path: Path, max_depth: int = 5, current_depth: int = 0):
    """Recursively explore directory structure"""
    if current_depth > max_depth:
        return
    
    if not path.exists():
        print(f"  [ERROR] Path does not exist: {path}")
        return
    
    indent = "  " * current_depth
    
    # Check for media files
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    video_files = []
    image_files = []
    
    try:
        for ext in video_exts:
            video_files.extend(list(path.glob(f"*{ext}")))
        for ext in image_exts:
            image_files.extend(list(path.glob(f"*{ext}")))
        
        # Count files in subdirectories (one level deep)
        subdir_videos = []
        subdir_images = []
        for subdir in path.iterdir():
            if subdir.is_dir():
                for ext in video_exts:
                    subdir_videos.extend(list(subdir.glob(f"*{ext}")))
                for ext in image_exts:
                    subdir_images.extend(list(subdir.glob(f"*{ext}")))
    except PermissionError:
        print(f"{indent}[PERMISSION DENIED] {path.name}")
        return
    
    total_files = len(video_files) + len(image_files) + len(subdir_videos) + len(subdir_images)
    
    if total_files > 0:
        print(f"{indent}[FOLDER] {path.name}/ ({total_files} files)")
        if video_files or subdir_videos:
            print(f"{indent}   Videos: {len(video_files) + len(subdir_videos)}")
        if image_files or subdir_images:
            print(f"{indent}   Images: {len(image_files) + len(subdir_images)}")
        
        # Show first few files
        all_files = video_files + image_files + subdir_videos + subdir_images
        for f in all_files[:5]:
            rel_path = f.relative_to(path)
            print(f"{indent}   - {rel_path}")
        if len(all_files) > 5:
            print(f"{indent}   ... and {len(all_files) - 5} more files")
    
    # Explore subdirectories
    try:
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        for subdir in sorted(subdirs):
            explore_directory(subdir, max_depth, current_depth + 1)
    except PermissionError:
        pass


def main():
    """Main diagnostic function"""
    print("="*70)
    print("Dataset Structure Diagnostic Tool")
    print("="*70)
    
    # Check configured dataset path
    from config.config import DATASET_PATH
    
    print(f"\n[INFO] Configured Dataset Path:")
    print(f"   {DATASET_PATH}")
    print(f"   Exists: {DATASET_PATH.exists()}")
    
    if not DATASET_PATH.exists():
        print(f"\n[ERROR] Dataset path does not exist!")
        print(f"\nPlease check config/config.py and update DATASET_PATH")
        return
    
    # Explore the directory
    print(f"\n[SEARCH] Exploring directory structure (max depth: 5)...")
    print("-"*70)
    explore_directory(DATASET_PATH, max_depth=5)
    
    # Check for common alternative locations
    print(f"\n\n[SEARCH] Checking for common alternative locations...")
    print("-"*70)
    
    base_path = DATASET_PATH.parent
    alternative_paths = [
        base_path / "ISL custom Data",
        base_path.parent / "ISL custom Data",
        Path(r"C:\Users\Aakash\Downloads\ISL Hand Gesture Dataset") / "ISL custom Data",
        Path(r"C:\Users\Aakash\Downloads\ISL Hand Gesture Dataset\ISL_Dataset_Extracted"),
    ]
    
    for alt_path in alternative_paths:
        if alt_path.exists() and alt_path != DATASET_PATH:
            print(f"\n[FOUND] Alternative path: {alt_path}")
            # Quick check for files
            video_count = len(list(alt_path.rglob("*.mp4"))) + len(list(alt_path.rglob("*.MP4")))
            image_count = len(list(alt_path.rglob("*.jpg"))) + len(list(alt_path.rglob("*.JPG"))) + \
                         len(list(alt_path.rglob("*.png"))) + len(list(alt_path.rglob("*.PNG")))
            if video_count > 0 or image_count > 0:
                print(f"   [OK] Contains {video_count} videos and {image_count} images")
                print(f"   [TIP] Consider updating DATASET_PATH to: {alt_path}")
    
    # Summary
    print(f"\n\n" + "="*70)
    print("Summary")
    print("="*70)
    
    # Count all files in dataset
    video_exts = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']
    image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    
    total_videos = 0
    total_images = 0
    
    try:
        for ext in video_exts:
            total_videos += len(list(DATASET_PATH.rglob(ext)))
        for ext in image_exts:
            total_images += len(list(DATASET_PATH.rglob(ext)))
    except Exception as e:
        print(f"Error counting files: {e}")
    
    print(f"\nTotal files found in dataset:")
    print(f"  Videos: {total_videos}")
    print(f"  Images: {total_images}")
    print(f"  Total: {total_videos + total_images}")
    
    if total_videos == 0 and total_images == 0:
        print(f"\n[ERROR] No media files found in dataset!")
        print(f"\nPossible issues:")
        print(f"  1. Dataset not extracted from RAR file")
        print(f"  2. Dataset files in different location")
        print(f"  3. Dataset path is incorrect")
        print(f"\nNext steps:")
        rar_path = DATASET_PATH.parent.parent / "ISL Hand Gesture Dataset" / "ISL custom Data.rar"
        print(f"  1. Extract the RAR file: {rar_path}")
        print(f"  2. Update DATASET_PATH in config/config.py")
        print(f"  3. Run this diagnostic again")
    else:
        print(f"\n[SUCCESS] Files found! Dataset should be loadable.")
        print(f"\nGesture folders detected:")
        folders = [d for d in DATASET_PATH.iterdir() if d.is_dir()]
        for folder in sorted(folders):
            folder_videos = sum(len(list(folder.rglob(ext))) for ext in video_exts)
            folder_images = sum(len(list(folder.rglob(ext))) for ext in image_exts)
            if folder_videos > 0 or folder_images > 0:
                print(f"  [OK] {folder.name}: {folder_videos} videos, {folder_images} images")
    
    print(f"\n" + "="*70)


if __name__ == "__main__":
    main()

