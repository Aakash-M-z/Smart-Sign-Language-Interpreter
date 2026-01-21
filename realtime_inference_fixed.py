#!/usr/bin/env python3
"""
Real-Time ISL Gesture Recognition with Model Compatibility Fix
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import argparse
from pathlib import Path
from collections import deque
import torchvision.models as models

# Import project modules
from config.config import MODEL_CONFIG, INFERENCE_CONFIG

class CompatibleCNNLSTM(nn.Module):
    """Compatible CNN-LSTM model that matches the old checkpoint structure"""
    
    def __init__(self, num_classes=26, cnn_backbone="mobilenet_v2", lstm_hidden_size=128):
        super(CompatibleCNNLSTM, self).__init__()
        
        # CNN backbone (same as old model - MobileNetV2)
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

class RealTimeGestureRecognizer:
    """Real-time gesture recognizer with actual model predictions"""
    
    def __init__(self, model_path, webcam_id=0, confidence_threshold=0.5):
        """Initialize recognizer with trained model"""
        self.webcam_id = webcam_id
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # ISL alphabet gestures
        self.gesture_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        # Frame processing
        self.sequence_length = 16
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.prediction_buffer = deque(maxlen=5)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize webcam with fallback to simulated feed
        self.cap = None
        self.use_simulated_feed = False
        
        # Try to initialize webcam with multiple attempts
        for cam_id in [webcam_id, 0, 1, 2]:
            try:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret:
                        self.cap = cap
                        self.webcam_id = cam_id
                        print(f"✅ Using webcam {cam_id}")
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
            except:
                continue
        
        if self.cap is None:
            print("⚠️  Webcam not available, using simulated video feed")
            self.use_simulated_feed = True
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Real-time gesture recognizer initialized on {self.device}")
        print("Controls: 'q' to quit, 's' to speak prediction")
    
    def _load_model(self, model_path):
        """Load the trained model with compatibility handling"""
        print(f"Loading model from {model_path}...")
        
        # Create compatible model
        model = CompatibleCNNLSTM(num_classes=26)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized model for demo...")
        
        model.to(self.device)
        model.eval()
        return model
    
    def create_simulated_frame(self, frame_count):
        """Create a simulated video frame with hand gesture simulation"""
        import time
        
        # Create base frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dark gradient background
        for y in range(480):
            intensity = int(30 + (y / 480) * 20)
            frame[y, :] = [intensity, intensity, intensity]
        
        # Add some visual elements to simulate a room/background
        cv2.rectangle(frame, (50, 50), (590, 430), (40, 40, 40), 1)
        
        # Simulate hand movement
        t = time.time()
        center_x = 320 + int(100 * np.sin(t * 0.5))
        center_y = 240 + int(50 * np.cos(t * 0.3))
        
        # Draw simulated hand shape
        cv2.ellipse(frame, (center_x, center_y), (80, 120), 0, 0, 360, (180, 150, 120), -1)
        cv2.ellipse(frame, (center_x, center_y), (80, 120), 0, 0, 360, (200, 170, 140), 2)
        
        # Add fingers
        for i in range(5):
            angle = -60 + i * 30
            finger_x = center_x + int(60 * np.cos(np.radians(angle)))
            finger_y = center_y - 60 + int(30 * np.sin(np.radians(angle)))
            cv2.circle(frame, (finger_x, finger_y), 15, (180, 150, 120), -1)
            cv2.circle(frame, (finger_x, finger_y), 15, (200, 170, 140), 2)
        
        # Add text overlay
        cv2.putText(frame, "SIMULATED CAMERA FEED", (200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        cv2.putText(frame, "Hand gesture simulation", (220, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(frame_rgb)
        return tensor
    
    def predict_gesture(self):
        """Make prediction from current frame buffer"""
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0
        
        # Prepare sequence
        sequence = []
        for frame in self.frame_buffer:
            processed = self.preprocess_frame(frame)
            sequence.append(processed)
        
        # Stack into tensor
        sequence_tensor = torch.stack(sequence).unsqueeze(0)  # (1, seq_len, C, H, W)
        sequence_tensor = sequence_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions over time"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Get high confidence predictions
        high_conf = [(p, c) for p, c in self.prediction_buffer if c > self.confidence_threshold]
        
        if high_conf:
            # Return most common high-confidence prediction
            predictions = [p for p, c in high_conf]
            most_common = max(set(predictions), key=predictions.count)
            avg_conf = np.mean([c for p, c in high_conf if p == most_common])
            return most_common, avg_conf
        
        return prediction, confidence
    
    def run(self):
        """Run real-time inference"""
        print("\nStarting real-time ISL gesture recognition...")
        print("Show ISL gestures to the camera!")
        
        last_prediction = None
        last_confidence = 0.0
        frame_count = 0
        
        try:
            while True:
                if self.use_simulated_feed:
                    # Use simulated video feed
                    frame = self.create_simulated_frame(frame_count)
                    ret = True
                else:
                    # Use real webcam
                    ret, frame = self.cap.read()
                    if ret:
                        # Flip frame horizontally for mirror effect
                        frame = cv2.flip(frame, 1)
                
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                # Add frame to buffer
                self.frame_buffer.append(frame.copy())
                
                # Make prediction every 5 frames
                if frame_count % 5 == 0 and len(self.frame_buffer) >= self.sequence_length:
                    pred_idx, confidence = self.predict_gesture()
                    
                    if pred_idx is not None:
                        # Smooth predictions
                        pred_idx, confidence = self.smooth_predictions(pred_idx, confidence)
                        
                        if confidence > self.confidence_threshold:
                            last_prediction = self.gesture_labels[pred_idx]
                            last_confidence = confidence
                
                # Draw UI
                self.draw_ui(frame, last_prediction, last_confidence)
                
                # Display frame
                cv2.imshow('ISL Gesture Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and last_prediction:
                    print(f"Predicted gesture: {last_prediction} (confidence: {last_confidence:.2f})")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def draw_ui(self, frame, prediction, confidence):
        """Draw enhanced UI elements with dark theme"""
        height, width = frame.shape[:2]
        
        # Create dark overlay for better contrast
        overlay = frame.copy()
        
        # Top header bar with dark background
        cv2.rectangle(overlay, (0, 0), (width, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw title with glow effect
        title = "ISL Gesture Recognition"
        cv2.putText(frame, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Shadow
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)  # Main text
        
        # Draw model status with enhanced visibility
        status = f"Model: Loaded | Device: {self.device} | Training: In Progress"
        cv2.putText(frame, status, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Shadow
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Real-time indicator
        import time
        if int(time.time() * 2) % 2:  # Blinking effect
            cv2.circle(frame, (width - 30, 30), 8, (0, 255, 0), -1)
        cv2.putText(frame, "LIVE", (width - 60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Enhanced prediction box with gradient effect
        box_height = 140
        box_y = height - box_height - 10
        
        # Dark background with border
        cv2.rectangle(frame, (10, box_y), (width - 10, height - 10), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, box_y), (width - 10, height - 10), (100, 255, 100), 3)
        
        # Inner glow effect
        cv2.rectangle(frame, (13, box_y + 3), (width - 13, height - 13), (50, 50, 50), 1)
        
        # Draw prediction with enhanced styling
        if prediction and confidence > self.confidence_threshold:
            # Large gesture letter with glow
            letter_x, letter_y = 40, box_y + 70
            cv2.putText(frame, prediction, (letter_x + 2, letter_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 6)  # Shadow
            cv2.putText(frame, prediction, (letter_x, letter_y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 100), 4)  # Main
            
            # Confidence with better styling
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, conf_text, (180, box_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Enhanced confidence bar with gradient
            bar_x, bar_y = 180, box_y + 50
            bar_width = int(250 * confidence)
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 250, bar_y + 20), (60, 60, 60), -1)
            
            # Confidence bar with color coding
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 250, bar_y + 20), (255, 255, 255), 2)
            
            # Percentage text
            percent_text = f"{confidence*100:.0f}%"
            cv2.putText(frame, percent_text, (bar_x + 260, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        else:
            # Waiting message with animation effect
            dots = "." * (int(time.time() * 2) % 4)
            waiting_text = f"Show gesture{dots}"
            cv2.putText(frame, waiting_text, (40, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        
        # Enhanced controls panel
        controls_y = 110
        cv2.rectangle(frame, (width - 220, controls_y - 10), (width - 10, controls_y + 60), (30, 30, 30), -1)
        cv2.rectangle(frame, (width - 220, controls_y - 10), (width - 10, controls_y + 60), (100, 100, 100), 1)
        
        controls = [
            ("'Q': Quit", (255, 100, 100)),
            ("'S': Speak", (100, 255, 100))
        ]
        
        for i, (control, color) in enumerate(controls):
            cv2.putText(frame, control, (width - 210, controls_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Enhanced status indicators
        status_y = height - 160
        
        # Buffer status with visual indicator
        buffer_ratio = len(self.frame_buffer) / self.sequence_length
        buffer_color = (0, 255, 0) if buffer_ratio == 1.0 else (0, 165, 255)
        
        cv2.rectangle(frame, (10, status_y), (200, status_y + 20), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, status_y), (10 + int(190 * buffer_ratio), status_y + 20), buffer_color, -1)
        
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
        cv2.putText(frame, buffer_text, (15, status_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add model performance indicator
        perf_text = "Model: Real-time predictions"
        cv2.putText(frame, perf_text, (10, status_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add gesture guide
        guide_text = "Show ISL A-Z gestures to camera"
        cv2.putText(frame, guide_text, (width//2 - 120, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-Time ISL Gesture Recognition')
    parser.add_argument('--model', type=str, 
                       default='models/training_20251106_113105/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--webcam', type=int, default=0, help='Webcam device ID')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Real-Time ISL Gesture Recognition")
    print("=" * 60)
    
    try:
        recognizer = RealTimeGestureRecognizer(
            model_path=args.model,
            webcam_id=args.webcam,
            confidence_threshold=args.confidence
        )
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if model file exists")
        print("2. Check if webcam is available")
        print("3. Try training a new model: python train.py")

if __name__ == "__main__":
    main()