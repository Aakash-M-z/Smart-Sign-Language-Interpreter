"""
Preprocessing utilities for video frames and sequences
"""
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, List, Optional
import random


class FramePreprocessor:
    """Preprocess individual video frames"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        """
        Initialize frame preprocessor
        
        Args:
            target_size: Target frame size (height, width)
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
        
        # Standard ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame as numpy array (RGB, normalized)
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, self.target_size)
        
        # Convert to float and normalize
        frame = frame.astype(np.float32) / 255.0
        
        if self.normalize:
            frame = (frame - self.mean) / self.std
        
        return frame


class Augmentation:
    """Data augmentation utilities for video sequences"""
    
    def __init__(self, config: dict):
        """
        Initialize augmentation
        
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enable_augmentation", True)
        
    def apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Apply random rotation"""
        if not self.enabled:
            return frame
            
        angle = random.uniform(-self.config["rotation_range"], self.config["rotation_range"])
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return frame
    
    def apply_flip(self, frame: np.ndarray) -> np.ndarray:
        """Apply random horizontal flip"""
        if not self.enabled or not self.config.get("horizontal_flip", False):
            return frame
            
        if random.random() < 0.5:
            frame = cv2.flip(frame, 1)
        return frame
    
    def apply_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment"""
        if not self.enabled:
            return frame
            
        brightness_range = self.config.get("brightness_range", (0.8, 1.2))
        factor = random.uniform(brightness_range[0], brightness_range[1])
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
        return frame
    
    def apply_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply random contrast adjustment"""
        if not self.enabled:
            return frame
            
        contrast_range = self.config.get("contrast_range", (0.8, 1.2))
        factor = random.uniform(contrast_range[0], contrast_range[1])
        mean = frame.mean()
        frame = np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)
        return frame
    
    def apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Apply random zoom"""
        if not self.enabled:
            return frame
            
        zoom_range = self.config.get("zoom_range", (0.9, 1.1))
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        h, w = frame.shape[:2]
        
        # Calculate crop size
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Crop center
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        frame = frame[start_y:start_y + new_h, start_x:start_x + new_w]
        
        # Resize back to original size
        frame = cv2.resize(frame, (w, h))
        return frame
    
    def apply_gaussian_noise(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise"""
        if not self.enabled or not self.config.get("gaussian_noise", False):
            return frame
        
        noise_std = self.config.get("noise_std", 0.01)
        noise = np.random.normal(0, noise_std * 255, frame.shape).astype(np.float32)
        frame = frame.astype(np.float32) + noise
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    
    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all augmentations to a frame
        
        Args:
            frame: Input frame
            
        Returns:
            Augmented frame
        """
        if not self.enabled:
            return frame
        
        # Convert to uint8 for augmentation operations
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Apply augmentations in a more controlled order
        frame = self.apply_brightness(frame)
        frame = self.apply_contrast(frame)
        frame = self.apply_rotation(frame)
        if self.config.get("horizontal_flip", False):  # Only flip if enabled
            frame = self.apply_flip(frame)
        frame = self.apply_zoom(frame)
        frame = self.apply_gaussian_noise(frame)
        
        # Convert back to float
        frame = frame.astype(np.float32) / 255.0
        
        return frame


def extract_frames_from_video(video_path: str, num_frames: int = 16, 
                              target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract frames from a video file with improved sampling
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target frame size (height, width)
        
    Returns:
        Array of frames shape (num_frames, height, width, channels)
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.array([])
    
    # Improved frame sampling: use uniform sampling but ensure we get meaningful frames
    if total_frames >= num_frames:
        # Uniform sampling across the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # If video has fewer frames, use all frames and repeat
        indices = list(range(total_frames))
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx % total_frames)
        ret, frame = cap.read()
        if ret:
            # Ensure frame is valid
            if frame is not None and frame.size > 0:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)  # Better interpolation
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    
    cap.release()
    
    # Pad or repeat frames if needed
    if len(frames) < num_frames:
        if len(frames) == 0:
            # If no frames extracted, return zeros
            return np.zeros((num_frames, *target_size, 3), dtype=np.uint8)
        # Repeat last frame or interpolate
        last_frame = frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8)
        while len(frames) < num_frames:
            frames.append(last_frame.copy())
    
    return np.array(frames[:num_frames])


def extract_frames_from_images(image_paths: List[str], num_frames: int = 16,
                               target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract frames from a list of image paths
    
    Args:
        image_paths: List of image file paths
        num_frames: Number of frames to extract
        target_size: Target frame size (height, width)
        
    Returns:
        Array of frames shape (num_frames, height, width, channels)
    """
    frames = []
    
    if len(image_paths) == 0:
        return np.array([])
    
    # Sample evenly from image paths
    if len(image_paths) >= num_frames:
        indices = np.linspace(0, len(image_paths) - 1, num_frames, dtype=int)
    else:
        indices = list(range(len(image_paths)))
    
    for idx in indices:
        if idx < len(image_paths):
            frame = cv2.imread(str(image_paths[idx]))
            if frame is not None:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    
    # Pad if needed
    if len(frames) < num_frames:
        last_frame = frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8)
        while len(frames) < num_frames:
            frames.append(last_frame)
    
    return np.array(frames[:num_frames])

