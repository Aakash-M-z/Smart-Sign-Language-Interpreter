#!/usr/bin/env python3
"""
Quick Demo - ISL Gesture Recognition
Creates a simple demo that shows the interface working with random predictions
"""

import cv2
import numpy as np
import torch
import random
import time
from pathlib import Path
import argparse
from collections import deque

# Import project modules
from config.config import MODEL_CONFIG

class DemoGestureRecognizer:
    """Demo gesture recognizer with random predictions"""
    
    def __init__(self, webcam_id=0, confidence_threshold=0.5):
        """Initialize demo recognizer"""
        self.webcam_id = webcam_id
        self.confidence_threshold = confidence_threshold
        
        # ISL alphabet gestures (A-Z)
        self.gesture_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.num_classes = len(self.gesture_labels)
        
        # Frame buffer for sequence
        self.sequence_length = MODEL_CONFIG["sequence_length"]
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        
        # Initialize webcam with fallback to simulated feed
        self.cap = None
        self.use_simulated_feed = False
        
        try:
            self.cap = cv2.VideoCapture(self.webcam_id)
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.cap = None
        except:
            self.cap = None
        
        if self.cap is None:
            print("⚠️  Webcam not available, using simulated video feed")
            self.use_simulated_feed = True
        else:
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Webcam initialized successfully")
        
        print("Demo Gesture Recognizer initialized!")
        print("This is a DEMO version with random predictions.")
        print("Train the actual model using: python train.py")
        print("\nControls:")
        print("- 'q': Quit")
        print("- 's': Speak last prediction")
        print("- Space: Force new prediction")
        
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
    
    def get_demo_prediction(self):
        """Generate demo prediction with realistic confidence"""
        # Simulate more realistic predictions
        gesture_idx = random.randint(0, self.num_classes - 1)
        confidence = random.uniform(0.4, 0.95)  # Higher confidence range
        return gesture_idx, confidence
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions over time"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Get most common prediction with high confidence
        high_conf_predictions = [(p, c) for p, c in self.prediction_buffer if c > self.confidence_threshold]
        
        if high_conf_predictions:
            # Return most recent high-confidence prediction
            return high_conf_predictions[-1]
        else:
            return prediction, confidence
    
    def run(self):
        """Run real-time demo inference"""
        print("\nStarting demo inference...")
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
                
                # Add frame to buffer (simulate preprocessing)
                self.frame_buffer.append(frame)
                
                # Make prediction every few frames
                if frame_count % 10 == 0 and len(self.frame_buffer) >= self.sequence_length:
                    # Demo prediction
                    pred_idx, confidence = self.get_demo_prediction()
                    
                    # Smooth predictions
                    pred_idx, confidence = self.smooth_predictions(pred_idx, confidence)
                    
                    if confidence > self.confidence_threshold:
                        last_prediction = self.gesture_labels[pred_idx]
                        last_confidence = confidence
                
                # Draw UI
                self.draw_ui(frame, last_prediction, last_confidence)
                
                # Display frame
                cv2.imshow('ISL Gesture Recognition - DEMO', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and last_prediction:
                    print(f"Speaking: {last_prediction}")
                    # In real version, this would use TTS
                elif key == ord(' '):
                    # Force new prediction
                    pred_idx, confidence = self.get_demo_prediction()
                    if confidence > self.confidence_threshold:
                        last_prediction = self.gesture_labels[pred_idx]
                        last_confidence = confidence
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        
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
        cv2.rectangle(overlay, (0, 0), (width, 90), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw title with glow effect
        title = "ISL Gesture Recognition - DEMO"
        cv2.putText(frame, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Shadow
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)  # Main text
        
        # Draw demo notice with enhanced visibility
        demo_text = "DEMO MODE - Random Predictions"
        cv2.putText(frame, demo_text, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Shadow
        cv2.putText(frame, demo_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)  # Orange text
        
        # Enhanced prediction box with gradient effect
        box_height = 140
        box_y = height - box_height - 10
        
        # Dark background with border
        cv2.rectangle(frame, (10, box_y), (width - 10, height - 10), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, box_y), (width - 10, height - 10), (100, 255, 100), 3)
        
        # Inner glow effect
        cv2.rectangle(frame, (13, box_y + 3), (width - 13, height - 13), (50, 50, 50), 1)
        
        # Draw prediction with enhanced styling
        if prediction and confidence > 0.3:  # Lower threshold for demo
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
            import time
            dots = "." * (int(time.time() * 2) % 4)
            waiting_text = f"Show gesture{dots}"
            cv2.putText(frame, waiting_text, (40, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        
        # Enhanced controls panel
        controls_y = 100
        cv2.rectangle(frame, (width - 220, controls_y - 10), (width - 10, controls_y + 80), (30, 30, 30), -1)
        cv2.rectangle(frame, (width - 220, controls_y - 10), (width - 10, controls_y + 80), (100, 100, 100), 1)
        
        controls = [
            ("'Q': Quit", (255, 100, 100)),
            ("'S': Speak", (100, 255, 100)),
            ("Space: New", (100, 200, 255))
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
        
        # Add frame rate indicator
        fps_text = "FPS: ~30"
        cv2.putText(frame, fps_text, (10, status_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add gesture guide
        guide_text = "Show ISL A-Z gestures to camera"
        cv2.putText(frame, guide_text, (width//2 - 120, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ISL Gesture Recognition Demo')
    parser.add_argument('--webcam', type=int, default=0, help='Webcam device ID')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ISL Gesture Recognition - DEMO MODE")
    print("=" * 60)
    print("This is a demonstration of the interface.")
    print("Predictions are random - train the model for real recognition!")
    print("=" * 60)
    
    try:
        recognizer = DemoGestureRecognizer(
            webcam_id=args.webcam,
            confidence_threshold=args.confidence
        )
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if webcam is connected and not used by another app")
        print("2. Try different webcam ID: --webcam 1")
        print("3. Check if OpenCV is properly installed")


if __name__ == "__main__":
    main()