import torch
import torch.nn as nn
from model import LaneCuttingDetector

class TemporalLaneCuttingDetector(nn.Module):
    def __init__(self, base_model, sequence_length=5):
        super(TemporalLaneCuttingDetector, self).__init__()
        self.base_model = base_model
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, image_sequence):
        batch_size, seq_len, c, h, w = image_sequence.shape
        features = []
        
        for i in range(seq_len):
            frame = image_sequence[:, i, :, :, :]
            with torch.no_grad():
                frame_features = self.base_model.extract_features(frame)
            features.append(frame_features)
        
        features = torch.stack(features, dim=1)
        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]
        output = self.classifier(final_features)
        
        return output