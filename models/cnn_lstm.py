"""
CNN-LSTM Model Architecture for ISL Hand Gesture Recognition
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class CNNLSTM(nn.Module):
    """
    CNN-LSTM architecture for video-based gesture recognition
    
    Architecture:
    - CNN backbone (MobileNetV2 or ResNet18) extracts spatial features from each frame
    - LSTM processes temporal sequences of features
    - Fully connected layers for classification
    """
    
    def __init__(self,
                 num_classes: int,
                 cnn_backbone: str = "mobilenet_v2",
                 input_size: Tuple[int, int] = (224, 224),
                 sequence_length: int = 16,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 dropout_rate: float = 0.5,
                 pretrained: bool = True):
        """
        Initialize CNN-LSTM model
        
        Args:
            num_classes: Number of gesture classes
            cnn_backbone: CNN backbone architecture ('mobilenet_v2' or 'resnet18')
            input_size: Input frame size (height, width)
            sequence_length: Number of frames in sequence
            lstm_hidden_size: LSTM hidden state size
            lstm_num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            pretrained: Whether to use pretrained CNN weights
        """
        super(CNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.cnn_backbone_name = cnn_backbone
        
        # Load CNN backbone
        if cnn_backbone == "mobilenet_v2":
            self.cnn_backbone = models.mobilenet_v2(pretrained=pretrained)
            # Remove classifier
            self.cnn_backbone.classifier = nn.Identity()
            # Get feature dimension
            cnn_feature_dim = 1280
        elif cnn_backbone == "resnet18":
            self.cnn_backbone = models.resnet18(pretrained=pretrained)
            # Remove final fully connected layer
            self.cnn_backbone.fc = nn.Identity()
            # Get feature dimension
            cnn_feature_dim = 512
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # Freeze early layers for faster training (optional)
        # Uncomment to freeze:
        # for param in list(self.cnn_backbone.parameters())[:-10]:
        #     param.requires_grad = False
        
        # LSTM for temporal modeling (bidirectional for better temporal understanding)
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=True  # Changed to bidirectional for better performance
        )
        
        # Fully connected layers (adjusted for bidirectional LSTM)
        lstm_output_size = lstm_hidden_size * 2  # *2 because bidirectional
        self.fc1 = nn.Linear(lstm_output_size, 512)  # Increased from 256 to 512
        self.bn1 = nn.BatchNorm1d(512)  # Added batch normalization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)  # Added intermediate layer
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # Less dropout in second layer
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for CNN: (batch_size * sequence_length, channels, height, width)
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract features using CNN: (batch_size * seq_len, cnn_feature_dim)
        cnn_features = self.cnn_backbone(x)
        
        # Reshape back: (batch_size, sequence_length, cnn_feature_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # Process through LSTM: (batch_size, seq_len, lstm_hidden_size * 2) for bidirectional
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # Use the last LSTM output: (batch_size, lstm_hidden_size * 2)
        # For bidirectional, concatenate forward and backward hidden states
        lstm_features = lstm_out[:, -1, :]  # Last timestep
        
        # Classification head with improved architecture
        out = self.fc1(lstm_features)
        # BatchNorm: use eval() mode properly handles batch_size=1
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out
    
    def get_feature_extractor(self):
        """Get CNN feature extractor (for inference)"""
        return self.cnn_backbone
    
    def get_lstm_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM features (for analysis/debugging)
        
        Args:
            x: Input tensor
            
        Returns:
            LSTM features
        """
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(cnn_features)
        return lstm_out[:, -1, :]


class CNNLSTMInference:
    """Wrapper class for efficient inference"""
    
    def __init__(self, model: CNNLSTM, device: torch.device):
        """
        Initialize inference wrapper
        
        Args:
            model: Trained CNNLSTM model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def predict(self, sequence: torch.Tensor) -> Tuple[int, float]:
        """
        Predict gesture from sequence
        
        Args:
            sequence: Input sequence tensor (1, seq_len, C, H, W)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        sequence = sequence.to(self.device)
        if len(sequence.shape) == 4:
            sequence = sequence.unsqueeze(0)  # Add batch dimension
        
        outputs = self.model(sequence)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()


def create_model(config: dict, num_classes: int) -> CNNLSTM:
    """
    Create CNN-LSTM model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of classes
        
    Returns:
        CNNLSTM model instance
    """
    model = CNNLSTM(
        num_classes=num_classes,
        cnn_backbone=config["cnn_backbone"],
        input_size=config["input_size"],
        sequence_length=config["sequence_length"],
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_num_layers=config["lstm_num_layers"],
        dropout_rate=config["dropout_rate"],
        pretrained=True
    )
    
    # Initialize weights for better training
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1)
    
    # Apply initialization to LSTM and FC layers
    model.lstm.apply(init_weights)
    model.fc1.apply(init_weights)
    model.fc2.apply(init_weights)
    model.fc3.apply(init_weights)
    
    return model

