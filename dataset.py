import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import torchvision.transforms as T
from torchvision.transforms import functional as F
from model import get_transform

class LaneCuttingDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform if transform is not None else get_transform(train=not is_test)
        self.is_test = is_test
        
        # Find all image files
        self.image_paths = []
        self.xml_paths = [] if not is_test else None
        
        for recording_folder in os.listdir(root_dir):
            recording_path = os.path.join(root_dir, recording_folder)
            if not os.path.isdir(recording_path) or not recording_folder.startswith('REC_'):
                continue
                
            annotations_path = os.path.join(recording_path, "Annotations")
            if not os.path.exists(annotations_path):
                continue
                
            # Get all JPG files
            for file in os.listdir(annotations_path):
                if file.lower().endswith('.jpg'):
                    image_path = os.path.join(annotations_path, file)
                    if not is_test:
                        xml_path = os.path.join(annotations_path, file[:-4] + '.xml')
                        if os.path.exists(xml_path):
                            self.image_paths.append(image_path)
                            self.xml_paths.append(xml_path)
                        else:
                            logging.warning(f"No XML file found for {image_path}")
                    else:
                        self.image_paths.append(image_path)
        
        if is_test:
            logging.info(f"Found {len(self.image_paths)} test images in {root_dir}")
        else:
            logging.info(f"Found {len(self.image_paths)} training images with annotations in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # Get image size
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        for obj in root.findall('object'):
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Validate box coordinates
            if xmin >= xmax or ymin >= ymax:
                continue
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                continue
            if xmax - xmin < 10 or ymax - ymin < 10:  # Filter out too small boxes
                continue
                
            # Get label
            name = obj.find('name').text.lower()
            label = 1  # Default to vehicle
            if 'cutting' in name:
                label = 2  # Lane cutting vehicle
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        # If no valid boxes found, return dummy box and label
        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # Background class
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        return boxes, labels
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Get original image dimensions
        height, width = image.shape[:2]
        
        # Convert to PIL Image for transforms
        image = F.to_pil_image(image)
        
        # Apply transforms
        image = self.transform(image)
        
        if self.is_test:
            return image, image_path
        
        # Parse XML annotations
        xml_path = self.xml_paths[idx]
        boxes, labels = self.parse_xml(xml_path)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target