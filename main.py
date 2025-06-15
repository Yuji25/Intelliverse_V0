# Update these paths
import os
import cv2
import numpy as np
import pandas as pd
from model import LaneCuttingDetector
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
DATA_ROOT = './Dataset'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'val')
TEST_DIR = os.path.join(DATA_ROOT, 'test')
OUTPUT_FILE = 'lane_cutting_predictions.csv'

# Print CUDA information if available
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


model = LaneCuttingDetector(num_classes=3)  # background, vehicle, lane_cutting
model.to(device)  # Move model to GPU