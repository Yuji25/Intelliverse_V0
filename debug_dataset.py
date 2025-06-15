from dataset import LaneCuttingDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset
train_dataset = LaneCuttingDataset('./Dataset/train', transform=transform)

# Test a few samples
for i in range(min(5, len(train_dataset))):
    image, target = train_dataset[i]
    
    # Convert tensor to numpy for visualization
    img_np = image.numpy().transpose(1, 2, 0)
    
    # Print information
    print(f"Image ID: {target['image_id']}")
    print(f"Lane cutting: {'Yes' if target['lane_cutting'] == 1 else 'No'}")
    print(f"Number of boxes: {len(target['boxes'])}")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(img_np)
    
    # Draw bounding boxes
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.numpy()
        color = 'r' if label == 2 else 'g'  # Red for lane cutting, green for vehicles
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor=color, linewidth=2))
    
    plt.title(f"Image {i}: {'Lane Cutting' if target['lane_cutting'] == 1 else 'No Lane Cutting'}")
    plt.show()

print("Dataset visualization completed!")