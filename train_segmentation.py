"""
SegFormer B2 Optimized Training Script
Arch: nvidia/segformer-b2-finetuned-ade-512-512 (Head replaced for 10 classes)
Targets: IoU > 0.6, High Speed, Low VRAM
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from PIL import Image
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# ============================================================================
# Mask Conversion & Class Mapping
# ============================================================================
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

# Fast Lookup Table for mask conversion
mask_lut = np.zeros(10001, dtype=np.uint8)
for raw_v, new_v in value_map.items():
    mask_lut[raw_v] = new_v

# Balanced weights for stability and IoU (Rare classes like Logs=6, Rocks=7 get boost)
CLASS_WEIGHTS = torch.tensor([
    1.0, 3.0, 2.0, 0.5, 1.0, 
    1.5, 4.0, 2.5, 0.5, 0.3
], dtype=torch.float32)

def convert_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    return mask_lut[mask]

# ============================================================================
# Dataset
# ============================================================================
class SegformerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = [f for f in os.listdir(self.image_dir) if f.lower().endswith('.png')]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = convert_mask(mask_path)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()

        return image, mask

def get_train_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ============================================================================
# Metrics
# ============================================================================
def fast_iou(preds, labels, n_classes=10):
    preds = torch.argmax(preds, dim=1)
    iou_list = []
    for cls in range(n_classes):
        intersect = ((preds == cls) & (labels == cls)).sum().float()
        union = ((preds == cls) | (labels == cls)).sum().float()
        if union > 0:
            iou_list.append(intersect / union)
    return torch.mean(torch.stack(iou_list)) if iou_list else torch.tensor(0.0)

# ============================================================================
# Main Training
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Arch: SegFormer B0 (Hyper-Speed) | Accelerating on: {device}")

    # Hyperparameters
    size = 320 # Faster resolution, still enough for detail
    batch_size = 8 # B0 is tiny, we can double the batch size for speed
    lr = 1e-4 # B0 can handle a slightly higher LR
    epochs = 15

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset')
    train_dir = os.path.join(dist_dir, 'train')
    val_dir = os.path.join(dist_dir, 'val')
    test_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages')
    checkpoint_dir = os.path.join(script_dir, 'checkpoints_segformer')
    test_preds_dir = os.path.join(script_dir, 'test_predictions')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(test_preds_dir, exist_ok=True)

    # Loaders
    train_ds = SegformerDataset(train_dir, transform=get_train_transforms(size))
    val_ds = SegformerDataset(val_dir, transform=get_val_transforms(size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Model
    print("Loading SegFormer B0-ADE Base...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", # Use the MiT-B0 backbone directly for max speed
        num_labels=10, 
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Targets: Fast Training + > 0.6 mIoU | Checkpoints in {checkpoint_dir}")
    print("=" * 60)

    history = {'val_iou': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=imgs, labels=masks)
                loss = outputs.loss
                # If weights applied manually (Transformers SegFormer handles loss if labels provided, 
                # but we'll use our custom weighted loss for better control)
                logits = outputs.logits
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_iou = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(pixel_values=imgs)
                logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                val_iou.append(fast_iou(logits, masks))
        
        avg_iou = torch.mean(torch.stack(val_iou)).item()
        scheduler.step()
        
        print(f"   [RESULT] Loss: {epoch_loss/len(train_loader):.4f} | Val mIoU: {avg_iou:.4f}")
        history['val_iou'].append(avg_iou)

        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}_iou_{avg_iou:.4f}.pth"))
        if avg_iou == max(history['val_iou']):
            torch.save(model.state_dict(), os.path.join(script_dir, "best_segformer.pth"))

    # ============================================================================
    # TEST INFERENCE
    # ============================================================================
    print("\nStarting inference on 'Offroad_Segmentation_testImages'...")
    model.eval()
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg'))]
    test_transform = get_val_transforms(size)
    
    with torch.no_grad():
        for f_name in tqdm(test_files, desc="Inferring"):
            img_raw = cv2.imread(os.path.join(test_dir, f_name))
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]
            
            aug = test_transform(image=img_rgb)
            img_tensor = aug['image'].unsqueeze(0).to(device)
            
            outputs = model(pixel_values=img_tensor)
            logits = F.interpolate(outputs.logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            cv2.imwrite(os.path.join(test_preds_dir, f_name.replace('.jpg', '.png')), pred_mask)

    print(f"Complete! Best mIoU: {max(history['val_iou']):.4f}")
    print(f"Predictions in: {test_preds_dir}")

if __name__ == "__main__":
    main()
