"""
HAIL MARY: Maximum Domain Generalization Training Pass
Goal: Improve test-set mIoU from 0.18 to 0.40+ by:
  1. Unfreezing the last 8 blocks of DINOv2 backbone
  2. Heavy blur, color jitter, and spatial augmentation
  3. Cosine annealing LR schedule
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')

# Official 10-class mapping
value_map = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
n_classes = len(value_map)
class_names = ["Trees","Lush Bushes","Dry Grass","Dry Bushes","Ground Clutter","Flowers","Logs","Rocks","Landscape","Sky"]

# Class weights to prioritize rare classes (Trees, Logs, Rocks)
class_weights = torch.tensor([2.5, 2.5, 1.5, 2.0, 2.5, 3.0, 3.0, 2.5, 0.5, 0.5], device=device)

class HailMaryDataset(Dataset):
    def __init__(self, data_dir, size=448):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.size = size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Very aggressive augmentations
        self.color_jitter = transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=7, sigma=(0.5, 4.0))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask_raw = Image.open(os.path.join(self.masks_dir, data_id))

        arr = np.array(mask_raw)
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        for raw_val, new_val in value_map.items():
            new_arr[arr == raw_val] = new_val
        mask = Image.fromarray(new_arr).resize((self.size, self.size), Image.NEAREST)

        # Spatial augmentation
        if random.random() > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() > 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)

        # Random crop and resize (simulates different zoom levels / env scale)
        if random.random() > 0.4:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.size * 0.8), int(self.size * 0.8)))
            image = TF.resized_crop(image, i, j, h, w, (self.size, self.size))
            mask = TF.resized_crop(mask, i, j, h, w, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)

        # Color augmentation (makes model focus on SHAPE not COLOR)
        image = self.color_jitter(image)
        if random.random() > 0.4:
            image = self.gaussian_blur(image)
        if random.random() > 0.7:
            image = TF.grayscale(image, num_output_channels=3)  # Occasionally train in grayscale!

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return self.normalize(image), mask


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        ts = size // 14
        self.ts = ts
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.BatchNorm2d(256), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 1), nn.GELU()
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.ts, self.ts, C).permute(0, 3, 1, 2)
        return self.classifier(self.block2(self.block1(self.stem(x))))


def compute_iou(inter, union):
    valid = union > 0
    return (inter[valid] / union[valid]).mean().item() if valid.any() else 0.0


def main():
    SIZE = 448
    N_EPOCHS = 10
    BATCH_SIZE = 6
    UNFREEZE_BLOCKS = 8  # Unfreeze last 8 blocks of DINOv2 for strong adaptation

    print(f"HAIL MARY Training on {device}")
    print(f"  Size={SIZE}, Epochs={N_EPOCHS}, Unfreezing last {UNFREEZE_BLOCKS} backbone blocks")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    train_loader = DataLoader(HailMaryDataset(train_dir, SIZE), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(HailMaryDataset(val_dir, SIZE), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load DINOv2 backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    for p in backbone.blocks[-UNFREEZE_BLOCKS:].parameters():
        p.requires_grad = True
    backbone.train()

    # Load best head as starting point
    head = SegmentationHead(768, n_classes, SIZE).to(device)
    best_head = "segmentation_head_robust.pth" if os.path.exists("segmentation_head_robust.pth") else "segmentation_head_fast.pth"
    try:
        head.load_state_dict(torch.load(best_head, map_location=device, weights_only=False))
        print(f"  Loaded starting weights from: {best_head}")
    except:
        print("  Starting head from scratch.")
    head.train()

    optimizer = optim.AdamW([
        {'params': backbone.blocks[-UNFREEZE_BLOCKS:].parameters(), 'lr': 5e-7},  # Very cautious backbone LR
        {'params': head.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    best_val_miou = 0.0

    for epoch in range(N_EPOCHS):
        backbone.train()
        head.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Hail Mary Epoch {epoch+1}/{N_EPOCHS}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    feats_full = backbone.forward_features(imgs)
                feats = feats_full["x_norm_patchtokens"]
                logits = head(feats)
                out = F.interpolate(logits, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
                loss = criterion(out, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        backbone.eval()
        head.eval()
        val_inter = torch.zeros(n_classes, device=device)
        val_union = torch.zeros(n_classes, device=device)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = head(feats)
                    out = F.interpolate(logits, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
                pred = out.argmax(1)
                for c in range(n_classes):
                    val_inter[c] += ((pred == c) & (masks == c)).sum()
                    val_union[c] += ((pred == c) | (masks == c)).sum()

        val_miou = compute_iou(val_inter, val_union)
        print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val mIoU: {val_miou:.4f}")

        # Save best backbone + head combo
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(head.state_dict(), "segmentation_head_hailmary.pth")
            torch.save(backbone.state_dict(), "dino_backbone_hailmary.pth")
            print(f"  *** NEW BEST {val_miou:.4f} — Saved! ***")

    print(f"\n==========================================")
    print(f"HAIL MARY DONE. Best Val mIoU: {best_val_miou:.4f}")
    print(f"  Head  -> segmentation_head_hailmary.pth")
    print(f"  Dino  -> dino_backbone_hailmary.pth")
    print(f"==========================================")
    print("Now run: py -3.11 evaluate_test_set.py")

if __name__ == "__main__":
    main()
