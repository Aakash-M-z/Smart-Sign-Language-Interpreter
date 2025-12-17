"""
Dataset loader for ISL Hand Gesture Recognition
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import random
from collections import defaultdict

from .preprocessing import (
    FramePreprocessor, 
    Augmentation, 
    extract_frames_from_video,
    extract_frames_from_images
)


class ISLDataset(Dataset):
    """Dataset class for ISL Hand Gesture Recognition"""
    
    def __init__(self, 
                 data_path: Path,
                 sequence_length: int = 16,
                 input_size: Tuple[int, int] = (224, 224),
                 augmentation_config: Optional[dict] = None,
                 mode: str = "train",
                 video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv'),
                 image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Initialize ISL Dataset
        
        Args:
            data_path: Path to dataset directory (should contain subdirectories for each gesture)
            sequence_length: Number of frames per sequence
            input_size: Target frame size (height, width)
            augmentation_config: Configuration for data augmentation
            mode: Dataset mode ('train', 'val', 'test')
            video_extensions: Supported video file extensions
            image_extensions: Supported image file extensions
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.mode = mode
        self.video_extensions = video_extensions
        self.image_extensions = image_extensions
        
        # Initialize preprocessor and augmentation
        self.preprocessor = FramePreprocessor(target_size=input_size, normalize=True)
        self.augmentation = Augmentation(augmentation_config or {}) if mode == "train" else None
        
        # Load dataset
        self.samples = self._load_dataset()
        
        if len(self.samples) == 0:
            print(f"\n[ERROR] No samples found in {data_path}")
            print(f"[WARN] Warning: Empty dataset detected. Please check:")
            print(f"   1. Dataset path is correct in config/config.py")
            print(f"   2. Dataset structure contains gesture folders with video/image files")
            print(f"   3. File permissions allow reading the dataset")
            # Don't raise error immediately - let the calling code handle it
    
    def _find_gesture_folders_recursive(self, root_path: Path, max_depth: int = 3) -> List[Path]:
        """
        Recursively search for gesture folders containing video/image files
        
        Args:
            root_path: Root directory to search
            max_depth: Maximum depth to search (default: 3)
            
        Returns:
            List of gesture folder paths
        """
        gesture_folders = []
        
        def search_recursive(path: Path, depth: int = 0):
            if depth > max_depth:
                return
            
            if not path.exists() or not path.is_dir():
                return
            
            # Check if current folder contains video/image files (directly or in subdirectories)
            has_videos = False
            has_images = False
            
            # Check directly in this folder
            for ext in self.video_extensions:
                if len(list(path.glob(f"*{ext}"))) > 0 or len(list(path.glob(f"*{ext.upper()}"))) > 0:
                    has_videos = True
                    break
            
            for ext in self.image_extensions:
                if len(list(path.glob(f"*{ext}"))) > 0 or len(list(path.glob(f"*{ext.upper()}"))) > 0:
                    has_images = True
                    break
            
            # If this folder has files, it's a gesture folder
            if has_videos or has_images:
                gesture_folders.append(path)
                return  # Don't search deeper if files found here
            
            # Search in subfolders
            try:
                for subfolder in path.iterdir():
                    if subfolder.is_dir():
                        search_recursive(subfolder, depth + 1)
            except (PermissionError, OSError) as e:
                # Skip folders we can't access
                pass
        
        search_recursive(root_path)
        return gesture_folders
    
    def _load_dataset(self) -> List[Tuple[str, int]]:
        """
        Load dataset samples with recursive search and better error handling
        
        Returns:
            List of tuples (file_path, label_index)
        """
        samples = []
        label_to_idx = {}
        idx_to_label = {}
        
        print(f"\n{'='*60}")
        print(f"Loading dataset from: {self.data_path}")
        print(f"{'='*60}")
        
        if not self.data_path.exists():
            print(f"[ERROR] Dataset path does not exist: {self.data_path}")
            print(f"\nPlease check the DATASET_PATH in config/config.py")
            return []
        
        # First, try to find gesture folders directly
        gesture_folders = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        # Check if direct folders have files
        folders_with_files = []
        for folder in gesture_folders:
            # Check for videos
            has_videos = False
            for ext in self.video_extensions:
                if len(list(folder.glob(f"*{ext}"))) > 0 or len(list(folder.glob(f"*{ext.upper()}"))) > 0:
                    has_videos = True
                    break
            
            # Check for images
            has_images = False
            for ext in self.image_extensions:
                if len(list(folder.glob(f"*{ext}"))) > 0 or len(list(folder.glob(f"*{ext.upper()}"))) > 0:
                    has_images = True
                    break
            
            if has_videos or has_images:
                folders_with_files.append(folder)
        
        # If no folders with files found, try recursive search
        if not folders_with_files:
            print("[WARN] No gesture folders with files found in direct path.")
            print("   Searching recursively...")
            gesture_folders = self._find_gesture_folders_recursive(self.data_path)
            if gesture_folders:
                print(f"   [OK] Found {len(gesture_folders)} gesture folders recursively")
            else:
                print("   [ERROR] No gesture folders found recursively either.")
        else:
            gesture_folders = folders_with_files
        
        if not gesture_folders:
            print(f"\n[ERROR] No gesture samples found — please check dataset path or structure.")
            print(f"\nExpected structure:")
            print(f"  dataset_root/")
            print(f"    ├── Gesture1/")
            print(f"    │   ├── video1.mp4")
            print(f"    │   └── ...")
            print(f"    ├── Gesture2/")
            print(f"    │   ├── image1.jpg")
            print(f"    │   └── ...")
            print(f"    └── ...")
            print(f"\nOr nested structure:")
            print(f"  dataset_root/")
            print(f"    └── train/")
            print(f"        ├── Gesture1/")
            print(f"        └── Gesture2/")
            return []
        
        # Sort gesture folders
        gesture_folders = sorted(gesture_folders, key=lambda x: x.name)
        
        # Create label mapping
        for idx, folder in enumerate(gesture_folders):
            label = folder.name
            label_to_idx[label] = idx
            idx_to_label[idx] = label
        
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.num_classes = len(label_to_idx)
        
        print(f"\n[INFO] Detected {self.num_classes} gesture classes:")
        print(f"{'='*60}")
        
        # Collect all samples
        for folder in gesture_folders:
            label = folder.name
            label_idx = label_to_idx[label]
            
            # Find video files (including in subdirectories)
            video_files = []
            for ext in self.video_extensions:
                video_files.extend(list(folder.rglob(f"*{ext}")))
                video_files.extend(list(folder.rglob(f"*{ext.upper()}")))
            
            # Find image files (including in subdirectories)
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(list(folder.rglob(f"*{ext}")))
                image_files.extend(list(folder.rglob(f"*{ext.upper()}")))
            
            # Count samples
            video_count = len(video_files)
            image_count = len(image_files)
            total_count = video_count + image_count
            
            if total_count == 0:
                print(f"  [WARN] {label}: 0 samples (empty folder)")
                continue
            
            # Add video files
            for video_file in video_files:
                samples.append((str(video_file), label_idx, "video"))
            
            # Group images into sequences
            if image_files:
                # Sort images by name to maintain temporal order
                image_files_sorted = sorted(image_files, key=lambda x: x.name)
                
                # Create sequences from images
                # Each sequence should have sequence_length images
                # We'll create overlapping sequences for better coverage
                num_images = len(image_files_sorted)
                
                if num_images >= self.sequence_length:
                    # Create sequences with overlap
                    step_size = max(1, self.sequence_length // 2)  # 50% overlap
                    for start_idx in range(0, num_images - self.sequence_length + 1, step_size):
                        sequence_images = image_files_sorted[start_idx:start_idx + self.sequence_length]
                        if len(sequence_images) == self.sequence_length:
                            samples.append(([str(img) for img in sequence_images], label_idx, "images"))
                    
                    # Also add the last sequence if we haven't covered it
                    if (num_images - self.sequence_length) % step_size != 0:
                        last_sequence = image_files_sorted[-self.sequence_length:]
                        samples.append(([str(img) for img in last_sequence], label_idx, "images"))
                elif num_images >= self.sequence_length // 2:
                    # If we have at least half the required frames, use all images and pad
                    samples.append(([str(img) for img in image_files_sorted], label_idx, "images"))
                # If too few images, skip this class (or we could pad, but that's not ideal)
            
            # Count sequences created for this class
            class_sequences = sum(1 for s in samples if s[1] == label_idx)
            
            # Print class info
            sample_info = []
            if video_count > 0:
                sample_info.append(f"{video_count} video(s)")
            if image_count > 0:
                # Count sequences from images (not individual images)
                image_sequences = class_sequences - video_count
                if image_sequences > 0:
                    sample_info.append(f"{image_sequences} sequence(s) from {image_count} images")
            
            print(f"  [OK] {label}: {class_sequences} sequences ({', '.join(sample_info) if sample_info else 'N/A'})")
        
        print(f"{'='*60}")
        print(f"[SUCCESS] Loaded {self.num_classes} gesture classes with {len(samples)} total samples.")
        print(f"{'='*60}\n")
        
        return samples
    
    def _group_images(self, image_files: List[Path]) -> List[List[str]]:
        """
        Group images that might belong to the same sequence
        
        Args:
            image_files: List of image file paths
            
        Returns:
            List of image groups
        """
        # Simple grouping: if images have similar names or are in subfolders
        groups = defaultdict(list)
        
        for img_file in sorted(image_files):
            # Try to extract a common prefix
            name = img_file.stem
            # Remove common suffixes like numbers at the end
            base_name = name.rstrip('0123456789_')
            groups[base_name].append(str(img_file))
        
        # If grouping results in many small groups, return all images as one group
        if len(groups) > len(image_files) * 0.5:
            return [[str(img) for img in image_files]]
        
        return list(groups.values())
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence_tensor, label)
        """
        sample_path, label, sample_type = self.samples[idx]
        
        # Extract frames
        if sample_type == "video":
            frames = extract_frames_from_video(
                sample_path, 
                self.sequence_length, 
                self.input_size
            )
        else:  # images
            frames = extract_frames_from_images(
                sample_path,
                self.sequence_length,
                self.input_size
            )
        
        if len(frames) == 0:
            # Return zero frames if extraction failed
            frames = np.zeros((self.sequence_length, *self.input_size, 3), dtype=np.uint8)
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            # Preprocess
            frame = self.preprocessor.preprocess(frame)
            
            # Augment if training
            if self.augmentation is not None:
                frame = self.augmentation.augment_frame(frame)
            
            processed_frames.append(frame)
        
        # Convert to tensor: (sequence_length, channels, height, width)
        sequence = np.array(processed_frames)
        sequence = np.transpose(sequence, (0, 3, 1, 2))  # (seq_len, H, W, C) -> (seq_len, C, H, W)
        sequence = torch.FloatTensor(sequence)
        
        return sequence, label


def create_data_loaders(data_path: Path,
                       sequence_length: int = 16,
                       input_size: Tuple[int, int] = (224, 224),
                       batch_size: int = 8,
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       test_split: float = 0.15,
                       augmentation_config: Optional[dict] = None,
                       num_workers: int = 4,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data_path: Path to dataset directory
        sequence_length: Number of frames per sequence
        input_size: Target frame size
        batch_size: Batch size for data loaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        augmentation_config: Configuration for data augmentation
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_mapping)
        
    Raises:
        ValueError: If no samples are found in the dataset
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset to get label mapping
    full_dataset = ISLDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        input_size=input_size,
        augmentation_config={},  # No augmentation for getting structure
        mode="train"
    )
    
    # Check if dataset is empty
    if len(full_dataset.samples) == 0:
        raise ValueError(
            f"\n[ERROR] No gesture samples found — please check dataset path or structure.\n"
            f"   Dataset path: {data_path}\n"
            f"   Please ensure the dataset contains gesture folders with video/image files."
        )
    
    label_mapping = {
        "label_to_idx": full_dataset.label_to_idx,
        "idx_to_label": full_dataset.idx_to_label,
        "num_classes": full_dataset.num_classes
    }
    
    # Split dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = ISLDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        input_size=input_size,
        augmentation_config=augmentation_config,
        mode="train"
    )
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    
    val_dataset = ISLDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        input_size=input_size,
        augmentation_config=None,
        mode="val"
    )
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
    val_dataset.label_to_idx = label_mapping["label_to_idx"]
    val_dataset.idx_to_label = label_mapping["idx_to_label"]
    val_dataset.num_classes = label_mapping["num_classes"]
    
    test_dataset = ISLDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        input_size=input_size,
        augmentation_config=None,
        mode="test"
    )
    test_dataset.samples = [test_dataset.samples[i] for i in test_indices]
    test_dataset.label_to_idx = label_mapping["label_to_idx"]
    test_dataset.idx_to_label = label_mapping["idx_to_label"]
    test_dataset.num_classes = label_mapping["num_classes"]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, label_mapping

