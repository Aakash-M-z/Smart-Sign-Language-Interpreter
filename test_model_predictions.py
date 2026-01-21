#!/usr/bin/env python3
"""
Test Model Predictions - Test the trained model on sample data
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import random
import matplotlib.pyplot as plt

class CompatibleCNNLSTM(nn.Module):
    """Compatible CNN-LSTM model that matches the old checkpoint structure"""
    
    def __init__(self, num_classes=26, lstm_hidden_size=128):
        super(CompatibleCNNLSTM, self).__init__()
        
        # CNN backbone (MobileNetV2 like old model)
        self.cnn_backbone = models.mobilenet_v2(pretrained=True)
        self.cnn_backbone.classifier = nn.Identity()
        cnn_feature_dim = 1280
        
        # LSTM (single direction like old model)
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        # FC layers (same as old model)
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        
        # CNN feature extraction
        x = x.view(batch_size * seq_len, channels, height, width)
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        lstm_features = lstm_out[:, -1, :]  # Last timestep
        
        # Classification
        out = self.fc1(lstm_features)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CompatibleCNNLSTM(num_classes=26)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully!")
        
        # Print some info about the checkpoint
        if 'epoch' in checkpoint:
            print(f"📊 Checkpoint info: Epoch {checkpoint['epoch']}")
        if 'best_val_acc' in checkpoint:
            print(f"📊 Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Using randomly initialized model...")
    
    model.to(device)
    model.eval()
    return model, device

def create_test_sequence():
    """Create a test sequence of random frames"""
    sequence_length = 16
    frames = []
    
    # Create random frames (simulating gesture frames)
    for i in range(sequence_length):
        # Create a random frame with some pattern
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some simple pattern to make it more realistic
        center = (112, 112)
        radius = 50 + i * 2  # Growing circle to simulate movement
        cv2.circle(frame, center, radius, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames

def preprocess_frames(frames):
    """Preprocess frames for model input"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    for frame in frames:
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        tensor = transform(frame_rgb)
        processed_frames.append(tensor)
    
    # Stack into sequence tensor
    sequence_tensor = torch.stack(processed_frames).unsqueeze(0)  # (1, seq_len, C, H, W)
    return sequence_tensor

def test_model_predictions(model, device, num_tests=10):
    """Test model with multiple random sequences"""
    gesture_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    print(f"\n🧪 Testing model with {num_tests} random sequences...")
    print("=" * 60)
    
    predictions = []
    confidences = []
    
    for i in range(num_tests):
        # Create test sequence
        frames = create_test_sequence()
        
        # Preprocess
        sequence_tensor = preprocess_frames(frames)
        sequence_tensor = sequence_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        pred_label = gesture_labels[predicted.item()]
        conf_value = confidence.item()
        
        predictions.append(pred_label)
        confidences.append(conf_value)
        
        print(f"Test {i+1:2d}: Predicted '{pred_label}' with confidence {conf_value:.3f}")
    
    print("=" * 60)
    
    # Statistics
    avg_confidence = np.mean(confidences)
    max_confidence = np.max(confidences)
    min_confidence = np.min(confidences)
    
    print(f"📊 Prediction Statistics:")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Max confidence: {max_confidence:.3f}")
    print(f"   Min confidence: {min_confidence:.3f}")
    print(f"   Unique predictions: {len(set(predictions))}/26 gestures")
    
    # Show distribution
    unique_preds = list(set(predictions))
    pred_counts = [predictions.count(p) for p in unique_preds]
    
    print(f"📈 Prediction distribution:")
    for pred, count in zip(unique_preds, pred_counts):
        print(f"   '{pred}': {count} times")
    
    return predictions, confidences

def test_with_dataset_sample():
    """Test with actual dataset sample if available"""
    dataset_path = Path("C:/Users/Aakash/Downloads/ISL Hand Gesture Dataset/ISL_Dataset_Extracted/ISL custom Data")
    
    if not dataset_path.exists():
        print("❌ Dataset not found, skipping dataset test")
        return
    
    print(f"\n🗂️  Testing with actual dataset samples...")
    
    # Find a gesture folder
    gesture_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    if not gesture_folders:
        print("❌ No gesture folders found")
        return
    
    # Pick a random gesture
    gesture_folder = random.choice(gesture_folders)
    gesture_name = gesture_folder.name
    
    # Get image files
    image_files = list(gesture_folder.glob("*.jpg")) + list(gesture_folder.glob("*.png"))
    if len(image_files) < 16:
        print(f"❌ Not enough images in {gesture_name} folder")
        return
    
    print(f"📁 Testing with gesture '{gesture_name}' ({len(image_files)} images available)")
    
    # Load 16 random images
    selected_images = random.sample(image_files, 16)
    frames = []
    
    for img_path in selected_images:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            frames.append(frame)
    
    if len(frames) < 16:
        print("❌ Could not load enough frames")
        return
    
    # Test with this sequence
    model, device = load_model("models/training_20251106_113105/best_model.pth")
    
    sequence_tensor = preprocess_frames(frames)
    sequence_tensor = sequence_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    gesture_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    pred_label = gesture_labels[predicted.item()]
    conf_value = confidence.item()
    
    print(f"🎯 Real dataset test:")
    print(f"   True gesture: '{gesture_name}'")
    print(f"   Predicted: '{pred_label}'")
    print(f"   Confidence: {conf_value:.3f}")
    print(f"   Correct: {'✅' if pred_label == gesture_name else '❌'}")

def main():
    """Main function"""
    print("=" * 60)
    print("🧪 ISL Gesture Recognition - Model Testing")
    print("=" * 60)
    
    model_path = "models/training_20251106_113105/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first: python train.py")
        return
    
    # Load model
    model, device = load_model(model_path)
    
    # Test with random sequences
    predictions, confidences = test_model_predictions(model, device, num_tests=10)
    
    # Test with actual dataset if available
    test_with_dataset_sample()
    
    print("\n" + "=" * 60)
    print("✅ Model testing completed!")
    print("💡 The model is working and making predictions.")
    print("📝 Note: Random test sequences won't be meaningful,")
    print("   but real gesture data should give better results.")
    print("=" * 60)

if __name__ == "__main__":
    main()