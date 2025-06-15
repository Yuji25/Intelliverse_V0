import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import logging
import torchvision.transforms as T
from torchvision.transforms import functional as F

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
        transforms.append(T.RandomAdjustSharpness(sharpness_factor=2, p=0.5))
    
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

def get_model(num_classes=3):
    model = LaneCuttingDetector(num_classes=num_classes)
    return model

class LaneCuttingDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images, targets=None):
        if torch.is_tensor(images[0]):
            images = [img for img in images]
        
        logging.info(f"Forward pass - Input images shape: {[img.shape for img in images]}")
        
        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            
            logging.info(f"Loss dict: {loss_dict}")
            
            return loss_dict
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
                
            for i, pred in enumerate(predictions):
                logging.info(f"Prediction {i} - Boxes: {pred['boxes'].shape}, Scores: {pred['scores'].shape}, Labels: {pred['labels'].shape}")
                if len(pred['boxes']) > 0:
                    logging.info(f"Max score: {pred['scores'].max():.4f}, Min score: {pred['scores'].min():.4f}")
                    logging.info(f"Unique labels: {torch.unique(pred['labels'])}")
            
            return predictions
    
    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.model.eval()
        return self