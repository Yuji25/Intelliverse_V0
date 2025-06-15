import torch
import torch.nn as nn
from model import LaneCuttingDetector

class TemporalLaneCuttingDetector(nn.Module):
    def __init__(self, base_model, sequence_length=5):
        super(TemporalLaneCuttingDetector, self).__init__()
        self.base_model = base_model
        self.sequence_length = sequence_length
        
        # LSTM for temporal context
        self.lstm = nn.LSTM(
            input_size=256,  # Feature size from base model
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # Final classification layer
        self.classifier = nn.Linear(128, 2)  # Binary classification: lane cutting or not
        
    def forward(self, image_sequence):
        batch_size, seq_len, c, h, w = image_sequence.shape
        features = []
        
        # Extract features from each frame using base model
        for i in range(seq_len):
            frame = image_sequence[:, i, :, :, :]  # Extract frame i from all sequences
            with torch.no_grad():
                # Get features from base model
                frame_features = self.base_model.extract_features(frame)
            features.append(frame_features)
        
        # Stack features and pass through LSTM
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, feature_dim]
        lstm_out, _ = self.lstm(features)
        
        # Use the last output from LSTM for classification
        final_features = lstm_out[:, -1, :]
        output = self.classifier(final_features)
        
        return output