# Real-Time ISL Hand Gesture Recognition System

A deep learning-based system for real-time recognition of Indian Sign Language (ISL) gestures using CNN-LSTM architecture, with text-to-speech output in regional languages.

## Features

- **CNN-LSTM Architecture**: Combines spatial feature extraction (CNN) with temporal pattern learning (LSTM)
- **Multiple CNN Backbones**: Supports MobileNetV2 and ResNet18
- **Real-Time Inference**: Webcam-based live gesture recognition
- **Regional Language Support**: Converts gestures to Hindi, Tamil, Telugu, and other regional languages
- **Text-to-Speech**: Speaks recognized gestures using gTTS or pyttsx3
- **Comprehensive Evaluation**: Accuracy, F1-score, precision, recall metrics
- **Data Augmentation**: Rotation, flip, brightness, contrast, zoom augmentations
- **Modular Design**: Well-organized codebase with separate modules for each component

## Project Structure

```
Smart Sign Language Interpreter/
├── config/
│   └── config.py          # Configuration settings
├── models/
│   └── cnn_lstm.py        # CNN-LSTM model architecture
├── utils/
│   ├── dataset.py         # Dataset loader
│   ├── preprocessing.py   # Frame preprocessing and augmentation
│   ├── evaluation.py      # Evaluation metrics and visualization
│   └── tts_engine.py      # Text-to-speech engine
├── data/                  # Dataset directory (created automatically)
├── models/                # Saved model checkpoints
├── logs/                  # Training logs and plots
├── train.py              # Training script
├── realtime_inference.py # Real-time inference script
└── requirements.txt       # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Webcam (for real-time inference)

### Setup

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Organize your ISL dataset in the following structure:
     ```
     dataset_root/
     ├── A/
     │   ├── video1.mp4
     │   ├── video2.mp4
     │   └── ...
     ├── B/
     │   ├── video1.mp4
     │   └── ...
     └── ...
     ```
   - Each folder should be named after the gesture label (e.g., "A", "B", "Hello", etc.)
   - Supports both video files (.mp4, .avi, .mov, .mkv) and image sequences (.jpg, .png)

4. **Update dataset path**:
   - Edit `config/config.py` and set `DATASET_PATH` to your dataset directory

## Usage

### Training

1. **Configure training parameters** (optional):
   - Edit `config/config.py` to adjust:
     - Model architecture (CNN backbone, LSTM parameters)
     - Training hyperparameters (batch size, learning rate, epochs)
     - Data augmentation settings

2. **Start training**:
   ```bash
   python train.py
   ```

   The training script will:
   - Load and split the dataset (70% train, 15% val, 15% test)
   - Train the CNN-LSTM model
   - Save the best model checkpoint
   - Generate evaluation metrics and plots
   - Save training history and configuration

3. **Training outputs**:
   - Model checkpoint: `models/training_YYYYMMDD_HHMMSS/best_model.pth`
   - Configuration: `models/training_YYYYMMDD_HHMMSS/config.json`
   - Training plots: `models/training_YYYYMMDD_HHMMSS/training_history.png`
   - Confusion matrix: `models/training_YYYYMMDD_HHMMSS/confusion_matrix.png`

### Real-Time Inference

1. **Run inference**:
   ```bash
   python realtime_inference.py --model path/to/best_model.pth
   ```

2. **Command-line options**:
   ```bash
   python realtime_inference.py \
       --model models/training_20241106_120000/best_model.pth \
       --webcam 0 \
       --language hi \
       --confidence 0.5 \
       --no-tts  # Disable TTS if needed
   ```

   Options:
   - `--model`: Path to trained model checkpoint (required)
   - `--config`: Path to config JSON (optional, auto-detected)
   - `--webcam`: Webcam device ID (default: 0)
   - `--language`: TTS language code - hi (Hindi), ta (Tamil), te (Telugu) (default: hi)
   - `--confidence`: Minimum confidence threshold (default: 0.5)
   - `--no-tts`: Disable text-to-speech output

3. **Controls**:
   - `q`: Quit the application
   - `s`: Speak the last recognized gesture

## Model Architecture

### CNN-LSTM Architecture

```
Input Video Sequence (16 frames × 224×224×3)
    ↓
CNN Backbone (MobileNetV2 or ResNet18)
    ↓
Spatial Features (per frame)
    ↓
LSTM (2 layers, 128 hidden units)
    ↓
Temporal Features
    ↓
Fully Connected Layers
    ↓
Gesture Classification
```

### Model Components

- **CNN Backbone**: Extracts spatial features from each frame
  - MobileNetV2: Lightweight, fast inference
  - ResNet18: More accurate, slightly slower
- **LSTM**: Learns temporal patterns across frames
- **Classification Head**: Fully connected layers for gesture classification

## Configuration

### Model Configuration

Edit `config/config.py` to customize:

```python
MODEL_CONFIG = {
    "cnn_backbone": "mobilenet_v2",  # or "resnet18"
    "input_size": (224, 224),
    "sequence_length": 16,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "dropout_rate": 0.5,
}
```

### Training Configuration

```python
TRAINING_CONFIG = {
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "patience": 10,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
}
```

### Data Augmentation

```python
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "horizontal_flip": True,
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    "zoom_range": (0.9, 1.1),
}
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1-scores
- **Precision**: Per-class and macro-averaged precision
- **Recall**: Per-class and macro-averaged recall
- **Confusion Matrix**: Visual representation of classification performance

## Regional Language Support

The system supports multiple regional languages:

- **Hindi (hi)**: अ, ब, स, ...
- **Tamil (ta)**: அ, ப, ச, ...
- **Telugu (te)**: అ, బ, స, ...

To add more languages, edit the `REGIONAL_MAPPINGS` dictionary in `utils/tts_engine.py`.

## Performance Optimization

### Training Speed

- Use GPU for training (CUDA)
- Adjust `num_workers` in training config based on CPU cores
- Reduce batch size if running out of memory
- Use MobileNetV2 for faster training

### Inference Speed

- Use MobileNetV2 backbone for faster inference
- Adjust `frame_skip` to process fewer frames
- Reduce `sequence_length` for shorter sequences
- Use GPU for inference if available

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size
   - Reduce sequence length
   - Use MobileNetV2 instead of ResNet18

2. **Webcam Not Found**:
   - Check webcam ID (try 0, 1, 2, etc.)
   - Ensure webcam is not being used by another application

3. **TTS Not Working**:
   - Install gTTS: `pip install gtts pygame`
   - Or install pyttsx3: `pip install pyttsx3`
   - Check internet connection (required for gTTS)

4. **Poor Recognition Accuracy**:
   - Ensure sufficient training data (minimum 50-100 samples per class)
   - Adjust augmentation parameters
   - Train for more epochs
   - Try different CNN backbone

## Future Enhancements

- [ ] Support for continuous gesture sequences
- [ ] Multi-hand gesture recognition
- [ ] Real-time gesture-to-text transcription
- [ ] Integration with NLP for sentence construction
- [ ] Mobile app deployment
- [ ] Web-based interface

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{isl_gesture_recognition,
  title = {Real-Time ISL Hand Gesture Recognition System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/isl-gesture-recognition}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV for computer vision utilities
- Contributors to the ISL dataset

