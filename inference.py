import torch
import torchvision.transforms as T
from model import get_model
from dataset import LaneCuttingDataset
import logging
import os
import cv2
import numpy as np
from pathlib import Path
import csv

logging.basicConfig(level=logging.INFO)

def visualize_detections(image, predictions, threshold=0.3):
    """Draw bounding boxes and labels on the image."""
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    # Colors for different classes (BGR format)
    colors = {
        1: (0, 255, 0),  # Green for non-cutting vehicles
        2: (0, 0, 255)   # Red for cutting vehicles
    }
    
    # Draw each detection
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            # Convert box coordinates to integers
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and score
            label_text = f"{'Cutting' if label == 2 else 'Vehicle'}: {score:.2f}"
            cv2.putText(image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = get_model(num_classes=3)
    
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    else:
        raise ValueError(f"No checkpoint found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'detections.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    
    test_dataset = LaneCuttingDataset('Dataset/Test', transform=None, is_test=True)
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, image_path = test_dataset[idx]
            frame_name = os.path.basename(image_path)
            
            logging.info(f"Processing image {idx+1}/{len(test_dataset)}: {image_path}")
            logging.info(f"Image tensor shape: {image.shape}, range: ({image.min()}, {image.max()})")
            
            image = image.unsqueeze(0)
            image = image.to(device)
            
            logging.info(f"Forward pass - Input images shape: {[img.shape for img in image]}")
            predictions = model(image)
            predictions = predictions[0]
            
            logging.info(f"Prediction 0 - Boxes: {predictions['boxes'].shape}, Scores: {predictions['scores'].shape}, Labels: {predictions['labels'].shape}")
            if len(predictions['boxes']) > 0:
                logging.info(f"Max score: {predictions['scores'].max():.4f}, Min score: {predictions['scores'].min():.4f}")
                logging.info(f"Unique labels: {torch.unique(predictions['labels'])}")
            
            mask = predictions['scores'] > 0.3
            filtered_boxes = predictions['boxes'][mask]
            filtered_scores = predictions['scores'][mask]
            filtered_labels = predictions['labels'][mask]
            
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                x1, y1, x2, y2 = box.cpu().numpy()
                csv_writer.writerow([frame_name, label.item(), score.item(), x1, y1, x2, y2])
            
            orig_image = cv2.imread(image_path)
            if orig_image is None:
                logging.error(f"Could not read image: {image_path}")
                continue
            
            output_image = visualize_detections(orig_image, predictions)
            
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, output_image)
            logging.info(f"Saved output to {output_path}")
            
            num_detections = len(filtered_boxes)
            logging.info(f"Found {num_detections} detections")
    
    csv_file.close()
    logging.info(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()