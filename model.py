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
    # Convert PIL image to tensor
    transforms.append(T.ToTensor())
    
    if train:
        # Add training transforms
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
        transforms.append(T.RandomAdjustSharpness(sharpness_factor=2, p=0.5))
    
    # Normalize
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

def get_model(num_classes=3):
    """Create and return a LaneCuttingDetector model instance."""
    model = LaneCuttingDetector(num_classes=num_classes)
    return model

class LaneCuttingDetector(nn.Module):
    def __init__(self, num_classes=3):  # Background + Vehicle + LaneCutting
        super(LaneCuttingDetector, self).__init__()
        
        # Load pre-trained model with proper weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            min_size=234,  # Match input image size
            max_size=416,  # Match input image size
            box_score_thresh=0.05,  # Lower confidence threshold
            rpn_pre_nms_top_n_train=2000,  # Increase number of region proposals
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=1000,
            rpn_post_nms_top_n_test=500,
            rpn_anchor_generator=None  # We'll define custom anchors
        )
        
        # Define custom anchor sizes and aspect ratios for vehicle detection
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.model.rpn.anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        # Replace the pre-trained head with a new one
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Initialize weights of the new head
        for name, param in self.model.roi_heads.box_predictor.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(self, images, targets=None):
        # Convert single tensor to list if needed
        if torch.is_tensor(images[0]):
            images = [img for img in images]
        
        # Add debugging information
        logging.info(f"Forward pass - Input images shape: {[img.shape for img in images]}")
        
        if self.training and targets is not None:
            # Training mode with targets
            loss_dict = self.model(images, targets)
            
            # Add debugging for losses
            logging.info(f"Loss dict: {loss_dict}")
            
            return loss_dict
        else:
            # Evaluation mode
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
                
            # Add debugging for predictions
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