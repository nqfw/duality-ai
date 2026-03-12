"""
Segmentation Training Script - Improved Version
Targets > 0.65 mIoU on validation set using DINOv2 + AdamW + Augmentations + Dice Loss
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
from tqdm import tqdm
import random

plt.switch_backend('Agg')

# ============================================================================
# Utility Functions
# ============================================================================
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

# ============================================================================
# Dataset with Augmentations
# ============================================================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, w, h, is_train=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.w = w
        self.h = h
        self.is_train = is_train

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        # Resize
        image = image.resize((self.w, self.h), Image.BILINEAR)
        mask = mask.resize((self.w, self.h), Image.NEAREST)

        # Augmentations
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Color Jitter
            if random.random() > 0.5:
                image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)(image)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)
        
        image = self.normalize(image)

        return image, mask

# ============================================================================
# Model: Improved Segmentation Head 
# ============================================================================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        # An expanded, deeper head with more capacity and dropout
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU(),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(0.2)
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)

# ============================================================================
# Custom Losses
# ============================================================================
def dice_loss(pred, target, smooth = 1.):
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ============================================================================
# Metrics
# ============================================================================
def compute_iou(pred, target, num_classes=10, ignore_index=255):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    iou_per_class = []
    
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    dice_per_class = []
    
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())
    return np.mean(dice_per_class)

def evaluate_metrics(model, backbone, data_loader, device, num_classes=10):
    iou_scores = []
    dice_scores = []
    model.eval()
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()
            iou_scores.append(compute_iou(outputs, labels, num_classes=num_classes))
            dice_scores.append(compute_dice(outputs, labels, num_classes=num_classes))
    
    model.train()
    return np.mean(iou_scores), np.mean(dice_scores)

# ============================================================================
# Training 
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters designed for ~0.65 mIoU bump
    batch_size = 4
    w = int(((960 / 2) // 14) * 14) # 476
    h = int(((540 / 2) // 14) * 14) # 266 
    
    lr = 5e-4
    n_epochs = 30 # increased epochs
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_improved')
    os.makedirs(output_dir, exist_ok=True)

    # Note dataset paths are fixed corresponding to where they actually unpacked
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir=data_dir, w=w, h=h, is_train=True)
    valset = MaskDataset(data_dir=val_dir, w=w, h=h, is_train=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    print("Loading DINOv2 backbone...")
    # Switched to base model for stronger representation capacity if VRAM allows
    backbone_name = "dinov2_vitb14_reg" 
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)

    imgs, _ = next(iter(train_loader))
    with torch.no_grad():
        output = backbone_model.forward_features(imgs.to(device))["x_norm_patchtokens"]
    n_embedding = output.shape[2]

    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    def combined_loss(pred, target):
        return ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    # Use AdamW + Cosine Annealing
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_iou = 0.0

    print("\nStarting enhanced training...")
    for epoch in range(n_epochs):
        classifier.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()

            loss = combined_loss(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()

        # Validation
        val_iou, val_dice = evaluate_metrics(classifier, backbone_model, val_loader, device, num_classes=n_classes)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {np.mean(train_losses):.4f} | Val mIoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_iou > best_iou:
            best_iou = val_iou
            model_path = os.path.join(script_dir, "segmentation_head_best.pth")
            torch.save(classifier.state_dict(), model_path)
            print(f"--> Saved new best model with mIoU: {best_iou:.4f}")

    print(f"\nTraining completed! Best Validation mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
