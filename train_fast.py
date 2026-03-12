"""
Fast Training Script - Target: 0.55-0.6 mIoU in < 1hr
Optimizations: 392x392 resolution, Frozen Backbone, Warm Start, AMP, Dice+CE Loss
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
from tqdm import tqdm
import random

# Device & Mixed Precision setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')

# Mapping for 10 classes
value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class FastDataset(Dataset):
    def __init__(self, data_dir, size=392, is_train=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.size = size
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask).resize((self.size, self.size), Image.NEAREST)

        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return self.normalize(image), mask

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.token_size = size // 14
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.BatchNorm2d(256), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 256, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.token_size, self.token_size, C).permute(0, 3, 1, 2)
        return self.classifier(self.block2(self.block1(self.stem(x))))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def dice_loss(pred, target):
    pred = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    inter = (pred * target_oh).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_oh.sum(dim=(2,3))
    return 1 - ((2. * inter + 1) / (union + 1)).mean()

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        if union > 0: ious.append((inter / union).item())
    return np.mean(ious)

def main():
    print(f"Starting FAST training on {device} (with Progressive Resizing)...")
    low_size = 308
    high_size = 476 # Original half-res
    n_epochs = 16
    switch_epoch = n_epochs // 2
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    # Start with low res
    current_size = low_size
    train_loader = DataLoader(FastDataset(train_dir, current_size, True), batch_size=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(FastDataset(val_dir, current_size, False), batch_size=8, shuffle=False)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device).eval()
    for p in backbone.parameters(): p.requires_grad = False

    model = SegmentationHead(768, n_classes, current_size).to(device)
    if os.path.exists("segmentation_head_best.pth"):
        print("Loading base weights for warm start...")
        model.load_state_dict(torch.load("segmentation_head_best.pth", map_location=device))

    focal_criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    for epoch in range(n_epochs):
        if epoch == switch_epoch:
            print(f"\nSwitching to high resolution: {high_size}x{high_size}...")
            current_size = high_size
            train_loader = DataLoader(FastDataset(train_dir, current_size, True), batch_size=4, shuffle=True, pin_memory=True)
            val_loader = DataLoader(FastDataset(val_dir, current_size, False), batch_size=4, shuffle=False)
            # Re-init model for new resolution (tokens size changes)
            new_model = SegmentationHead(768, n_classes, current_size).to(device)
            new_model.load_state_dict(model.state_dict())
            model = new_model
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) # Lower LR for fine-tuning

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} (Size: {current_size})")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            
            with torch.amp.autocast('cuda'):
                logits = model(feats)
                out = F.interpolate(logits, size=(current_size, current_size), mode='bilinear', align_corners=False)
                loss = focal_criterion(out, masks) + 0.5 * dice_loss(out, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                out = F.interpolate(model(feats), size=(current_size, current_size), mode='bilinear')
                ious.append(compute_iou(out, masks))
        
        print(f"Epoch {epoch+1} Val mIoU: {np.mean(ious):.4f}")
        torch.save(model.state_dict(), "segmentation_head_fast.pth")

if __name__ == "__main__": main()
