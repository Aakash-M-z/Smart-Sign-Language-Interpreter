"""
Real-time inference script for ISL Hand Gesture Recognition
"""
import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path
import json
import argparse
from typing import Optional

from config.config import INFERENCE_CONFIG, TTS_CONFIG, MODEL_CONFIG
from models.cnn_lstm import CNNLSTM, CNNLSTMInference
from utils.preprocessing import FramePreprocessor
from utils.tts_engine import TTSEngine, get_regional_text


class RealTimeGestureRecognizer:
    """Real-time gesture recognition system"""
    
    def __init__(self,
                 model_path: Path,
                 config_path: Optional[Path] = None,
                 webcam_id: int = 0,
                 frame_skip: int = 2,
                 confidence_threshold: float = 0.5,
                 smoothing_window: int = 5,
                 tts_enabled: bool = True,
                 tts_language: str = "hi"):
        """
        Initialize real-time gesture recognizer
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model config JSON
            webcam_id: Webcam device ID
            frame_skip: Process every Nth frame
            confidence_threshold: Minimum confidence for prediction
            smoothing_window: Number of predictions to average
            tts_enabled: Enable text-to-speech
            tts_language: TTS language code
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model configuration
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.model_config = config_data['model_config']
                self.label_mapping = config_data['label_mapping']
        else:
            # Use default config
            self.model_config = MODEL_CONFIG
            self.label_mapping = {"idx_to_label": {}, "label_to_idx": {}}
        
        # Load model
        self.model = self._load_model(model_path)
        self.inference_model = CNNLSTMInference(self.model, self.device)
        
        # Initialize frame buffer
        self.sequence_length = self.model_config["sequence_length"]
        self.input_size = self.model_config["input_size"]
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Initialize preprocessor
        self.preprocessor = FramePreprocessor(
            target_size=self.input_size,
            normalize=True
        )
        
        # Inference settings
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = min(smoothing_window, 3)  # Reduce smoothing window for faster response
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # TTS
        self.tts_enabled = tts_enabled
        if tts_enabled:
            self.tts = TTSEngine(
                engine=TTS_CONFIG["engine"],
                language=tts_language,
                slow=TTS_CONFIG["slow"]
            )
        else:
            self.tts = None
        
        self.last_prediction = None
        self.frame_count = 0
        
    def _load_model(self, model_path: Path) -> CNNLSTM:
        """Load trained model"""
        print(f"Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get number of classes from label mapping
        num_classes = self.label_mapping.get("num_classes", 
                                             len(self.label_mapping.get("idx_to_label", {})))
        
        if num_classes == 0:
            # Try to infer from checkpoint (check fc3 first for new architecture, then fc2 for old)
            model_state = checkpoint.get('model_state_dict', {})
            if 'fc3.weight' in model_state:
                num_classes = model_state['fc3.weight'].shape[0]
            elif 'fc2.weight' in model_state:
                num_classes = model_state['fc2.weight'].shape[0]
            else:
                raise ValueError("Could not determine num_classes from checkpoint")
        
        # Create model
        model = CNNLSTM(
            num_classes=num_classes,
            cnn_backbone=self.model_config["cnn_backbone"],
            input_size=tuple(self.model_config["input_size"]),
            sequence_length=self.model_config["sequence_length"],
            lstm_hidden_size=self.model_config["lstm_hidden_size"],
            lstm_num_layers=self.model_config["lstm_num_layers"],
            dropout_rate=self.model_config["dropout_rate"],
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"[OK] Model loaded successfully ({num_classes} classes)")
        return model
    
    def process_frame(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Process a single frame and return prediction if available
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Tuple of (predicted_label, confidence) or None
        """
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return None
        
        # Preprocess frame
        processed_frame = self.preprocessor.preprocess(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Check if we have enough frames
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        # Convert to tensor
        sequence = np.array(list(self.frame_buffer))
        sequence = np.transpose(sequence, (0, 3, 1, 2))  # (seq_len, H, W, C) -> (seq_len, C, H, W)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Predict
        try:
            predicted_idx, confidence = self.inference_model.predict(sequence_tensor)
            
            # Add to prediction buffer for smoothing (even if below threshold)
            self.prediction_buffer.append((predicted_idx, confidence))
            
            # Get most common prediction from buffer
            if len(self.prediction_buffer) >= self.smoothing_window:
                predictions = [pred[0] for pred in self.prediction_buffer]
                avg_confidence = np.mean([pred[1] for pred in self.prediction_buffer])
                
                # Get most frequent prediction
                from collections import Counter
                pred_counter = Counter(predictions)
                smoothed_pred = pred_counter.most_common(1)[0][0]
                
                # Return prediction even if below threshold (we'll show it with different color)
                return smoothed_pred, avg_confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
        
        return None
    
    def get_label_text(self, label_idx: int) -> str:
        """Get label text from index"""
        idx_to_label = self.label_mapping.get("idx_to_label", {})
        return idx_to_label.get(str(label_idx), idx_to_label.get(label_idx, f"Class {label_idx}"))
    
    def run(self):
        """Run real-time inference loop"""
        # Initialize webcam
        cap = cv2.VideoCapture(INFERENCE_CONFIG["webcam_id"])
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam {INFERENCE_CONFIG['webcam_id']}")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, INFERENCE_CONFIG.get("fps", 60))
        
        print("\n" + "="*60)
        print("Real-Time ISL Gesture Recognition")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 's' to speak last prediction")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            prediction = self.process_frame(frame)
            
            # Display prediction
            if prediction:
                label_idx, confidence = prediction
                label_text = self.get_label_text(label_idx)
                regional_text = get_regional_text(label_text, TTS_CONFIG["language"])
                
                self.last_prediction = (label_text, regional_text, confidence)
                
                # Display on frame
                text = f"Gesture: {label_text} ({regional_text})"
                confidence_text = f"Confidence: {confidence:.2f}"
                
                # Color based on confidence
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255) if confidence > 0.3 else (0, 0, 255)
                
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, confidence_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Print to console for debugging
                print(f"Prediction: {label_text} (Confidence: {confidence:.3f})")
                
                # Speak prediction (only if it's a new prediction)
                if self.tts_enabled and self.tts:
                    # Only speak if confidence is high and it's a new prediction
                    if confidence > 0.7 and (not hasattr(self, '_last_spoken') or 
                                            self._last_spoken != label_text):
                        try:
                            self.tts.speak(regional_text)
                            self._last_spoken = label_text
                        except Exception as e:
                            print(f"TTS error: {e}")
            else:
                # Show status when buffer is filling or no prediction
                if len(self.frame_buffer) < self.sequence_length:
                    status_text = f"Buffering frames... {len(self.frame_buffer)}/{self.sequence_length}"
                    cv2.putText(frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                elif len(self.prediction_buffer) < self.smoothing_window:
                    status_text = f"Collecting predictions... {len(self.prediction_buffer)}/{self.smoothing_window}"
                    cv2.putText(frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    status_text = "Waiting for gesture..."
                    cv2.putText(frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Display frame buffer status
            buffer_status = f"Frames: {len(self.frame_buffer)}/{self.sequence_length}"
            cv2.putText(frame, buffer_status, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to speak", 
                       (10, frame.shape[0] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('ISL Gesture Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and self.last_prediction:
                # Speak last prediction
                if self.tts:
                    try:
                        self.tts.speak(self.last_prediction[1])
                    except Exception as e:
                        print(f"TTS error: {e}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.tts:
            self.tts.stop()
        print("\n[OK] Inference stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time ISL Gesture Recognition')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config JSON file')
    parser.add_argument('--webcam', type=int, default=0,
                       help='Webcam device ID')
    parser.add_argument('--no-tts', action='store_true',
                       help='Disable text-to-speech')
    parser.add_argument('--language', type=str, default='hi',
                       help='TTS language code (hi, ta, te, etc.)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Find config file if not provided
    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        # Try to find config in model directory
        model_path = Path(args.model)
        config_path = model_path.parent / 'config.json'
        if not config_path.exists():
            config_path = None
    
    # Create recognizer
    recognizer = RealTimeGestureRecognizer(
        model_path=Path(args.model),
        config_path=config_path,
        webcam_id=args.webcam,
        tts_enabled=not args.no_tts,
        tts_language=args.language,
        confidence_threshold=args.confidence
    )
    
    # Run
    recognizer.run()


if __name__ == "__main__":
    main()

