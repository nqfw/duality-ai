"""
Robustness Fine-tuning Script
Goal: Improve generalizability for unseen locations using heavy augmentation.
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
import random
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')

value_map = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
n_classes = len(value_map)

class RobustDataset(Dataset):
    def __init__(self, data_dir, size=448, is_train=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.size = size
        self.is_train = is_train
        
        # Base normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Heavy augmentations for domain shift
        if is_train:
            self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask_raw = Image.open(os.path.join(self.masks_dir, data_id))
        
        # Efficient Mask Conversion
        arr = np.array(mask_raw)
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        for raw_val, new_val in value_map.items():
            new_arr[arr == raw_val] = new_val
        mask = Image.fromarray(new_arr).resize((self.size, self.size), Image.NEAREST)

        if self.is_train:
            # 1. Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 2. Color Jitter (Lighting variations)
            image = self.color_jitter(image)
            
            # 3. Blur (Simulate camera focus/dust)
            if random.random() > 0.3:
                image = self.gaussian_blur(image)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return self.normalize(image), mask

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.ts = size // 14
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.BatchNorm2d(256), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 256, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.ts, self.ts, C).permute(0, 3, 1, 2)
        return self.classifier(self.block2(self.block1(self.stem(x))))

def main():
    print(f"Starting Domain Robustness Pass on {device}...")
    size = 448
    n_epochs = 5 # Short, intense pass
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    train_loader = DataLoader(RobustDataset(train_dir, size, True), batch_size=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(RobustDataset(val_dir, size, False), batch_size=8, shuffle=False)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device)
    # Unfreeze the last 2 blocks of the backbone for adaptation
    for p in backbone.parameters(): p.requires_grad = False
    for p in backbone.blocks[-2:].parameters(): p.requires_grad = True
    backbone.train()

    model = SegmentationHead(768, n_classes, size).to(device)
    model.load_state_dict(torch.load("segmentation_head_fast.pth", map_location=device, weights_only=False))
    model.train()

    criterion = nn.CrossEntropyLoss() # Standard for stability now
    optimizer = optim.AdamW([
        {'params': backbone.blocks[-2:].parameters(), 'lr': 1e-6}, # Very small LR for backbone
        {'params': model.parameters(), 'lr': 5e-5}
    ])
    
    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, desc=f"Robustness Epoch {epoch+1}/{n_epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda'):
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(feats)
                out = F.interpolate(logits, size=(size, size), mode='bilinear', align_corners=False)
                loss = F.cross_entropy(out, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), "segmentation_head_robust.pth")
        torch.save(backbone.state_dict(), "dino_backbone_robust.pth")

if __name__ == "__main__": main()
