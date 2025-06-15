import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import get_model
from dataset import LaneCuttingDataset
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        if all(t['labels'].sum() == 0 for t in targets):
            continue
            
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        try:
            loss_dict = model.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if torch.isnan(losses):
                logging.warning("Skipping batch due to nan loss")
                continue
            
            losses.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += losses.item()
            
            pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})
            
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / len(data_loader)

def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Validation')
        for images, targets in pbar:
            if all(t['labels'].sum() == 0 for t in targets):
                continue
                
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            try:
                model.train()
                loss_dict = model.model(images, targets)
                model.eval()
                
                losses = sum(loss for loss in loss_dict.values())
                
                if not torch.isnan(losses):
                    total_loss += losses.item()
                    num_batches += 1
                    pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})
            except Exception as e:
                logging.error(f"Error in validation: {str(e)}")
                continue
    
    return total_loss / max(num_batches, 1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    train_dataset = LaneCuttingDataset('Dataset/Train', transform=None, is_test=False)
    val_dataset = LaneCuttingDataset('Dataset/Val', transform=None, is_test=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model = get_model(num_classes=3)
    model = model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.0001)
    
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    num_epochs = 1
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch+1)
        logging.info(f"Training loss: {train_loss:.4f}")
        
        val_loss = validate(model, val_loader, device)
        logging.info(f"Validation loss: {val_loss:.4f}")
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pth')
            logging.info("Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'val_loss': val_loss,
        }, 'checkpoints/latest_model.pth')
        
        logging.info(f"Completed epoch {epoch+1}")

if __name__ == "__main__":
    main()