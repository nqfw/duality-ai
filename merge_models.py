"""
Model Ensemble Script - Target: 5-7% mIoU increase
Combines DINOv2 (0.51 mIoU) and Segformer B2 (0.59 mIoU) using Weighted Averaging + TTA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Architecture Definitions
# ==========================================

# DINOv2 Head (Match train_fast.py)
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tw, th):
        super().__init__()
        self.tw = tw
        self.th = th
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.BatchNorm2d(256), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 256, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.th, self.tw, C).permute(0, 3, 1, 2)
        return self.classifier(self.block2(self.block1(self.stem(x))))

# ==========================================
# 2. Data Loading
# ==========================================
# Strict 10-class mapping from Duality AI PDF
value_map = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape
    10000: 9  # Sky
}
n_classes = len(value_map)
class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"]

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class EvalDataset(Dataset):
    def __init__(self, data_dir, w=448, h=252):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.w = w
        self.h = h
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB").resize((self.w, self.h), Image.BILINEAR)
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask).resize((self.w, self.h), Image.NEAREST)
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return self.normalize(image), mask

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        if union > 0: ious.append((inter / union).item())
    return np.mean(ious)

# ==========================================
# 3. Ensemble Inference Logic
# ==========================================

def ensemble_inference(image, dino_backbone, dino_head, segformer, weights=[0.75, 0.25]):
    """
    weights[0]: DINOv2 weight (0.51 mIoU)
    weights[1]: Segformer weight (0.59 mIoU)
    Includes TTA (Horizontal Flip)
    """
    # Create a batch of [Original, Flipped]
    batch_imgs = torch.stack([image, TF.hflip(image)]) 
    
    with torch.no_grad():
        # DINOv2 Predictions
        feats = dino_backbone.forward_features(batch_imgs)["x_norm_patchtokens"]
        logits_dino = dino_head(feats)
        # Using h and w from outer scope if needed, but here we use the image shape
        h_img, w_img = image.shape[-2], image.shape[-1]
        logits_dino = F.interpolate(logits_dino, size=(h_img, w_img), mode='bilinear', align_corners=False)
        
        # Segformer Predictions
        outputs = segformer(batch_imgs)
        logits_seg = F.interpolate(outputs.logits, size=(h_img, w_img), mode='bilinear', align_corners=False)

    # Reverse TTA for flipped images
    logits_dino[1] = TF.hflip(logits_dino[1])
    logits_seg[1] = TF.hflip(logits_seg[1])

    # Convert to Probabilities (Softmax) before blending for stability
    prob_dino = F.softmax(logits_dino, dim=1)
    prob_seg = F.softmax(logits_seg, dim=1)

    # Average TTA results for each model
    final_prob_dino = (prob_dino[0] + prob_dino[1]) / 2.0
    final_prob_seg = (prob_seg[0] + prob_seg[1]) / 2.0

    # Weighted Blend of Probabilities
    # DINO is strong on individual classes, Segformer on structure.
    # 75% DINOv2 + 25% Segformer blend
    blended_probs = (weights[0] * final_prob_dino) + (weights[1] * final_prob_seg)

    return blended_probs.unsqueeze(0)

def main():
    print(f"Initializing Comprehensive Ensemble Evaluation on {device}...")
    # Bumping resolution to 784x448 (Multiple of 14 and 4)
    # Higher resolution usually helps Segformer significantly.
    w, h = 784, 448 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')
    val_loader = DataLoader(EvalDataset(val_dir, w, h), batch_size=1, shuffle=False)

    # Load Model 1: DINOv2
    print("Loading DINOv2 + Head...")
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device).eval()
    dino_head = SegmentationHead(768, n_classes, w // 14, h // 14).to(device).eval()
    if os.path.exists("segmentation_head_fast.pth"):
        dino_head.load_state_dict(torch.load("segmentation_head_fast.pth", map_location=device, weights_only=False))
    
    # Load Model 2: Segformer B2
    print("Loading Segformer B2...")
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024", 
        num_labels=n_classes, 
        ignore_mismatched_sizes=True
    ).to(device)
    
    if os.path.exists("best_b2.pth"):
        print("Loading friend's Segformer weights...")
        checkpoint = torch.load("best_b2.pth", map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not any(k.startswith("model.") for k in segformer.state_dict().keys()) and any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        segformer.load_state_dict(state_dict, strict=False)
    
    segformer.eval()

    # Intersections and Unions for global mIoU
    # [0] = DINO, [1] = Segformer, [2] = Ensemble
    inter = torch.zeros((3, n_classes), device=device)
    union = torch.zeros((3, n_classes), device=device)
    
    print(f"\nEvaluating on {len(val_loader)} images at {w}x{h}...")
    for img_tensor, mask in tqdm(val_loader):
        img_tensor, mask = img_tensor.to(device).squeeze(0), mask.to(device)
        target = mask.view(-1)
        
        # 1. Individual DINOv2
        with torch.no_grad():
            feat = dino_backbone.forward_features(img_tensor.unsqueeze(0))["x_norm_patchtokens"]
            out_dino = F.interpolate(dino_head(feat), size=(h, w), mode='bilinear', align_corners=False)
            pred_dino = torch.argmax(out_dino, dim=1).view(-1)
            for c in range(n_classes):
                inter[0, c] += ((pred_dino == c) & (target == c)).sum()
                union[0, c] += ((pred_dino == c) | (target == c)).sum()
            
        # 2. Individual Segformer
        with torch.no_grad():
            outputs = segformer(img_tensor.unsqueeze(0))
            out_seg = F.interpolate(outputs.logits, size=(h, w), mode='bilinear', align_corners=False)
            pred_seg = torch.argmax(out_seg, dim=1).view(-1)
            for c in range(n_classes):
                inter[1, c] += ((pred_seg == c) & (target == c)).sum()
                union[1, c] += ((pred_seg == c) | (target == c)).sum()

        # 3. Ensemble
        blended_out = ensemble_inference(img_tensor, dino_backbone, dino_head, segformer)
        pred_ens = torch.argmax(blended_out, dim=1).view(-1)
        for c in range(n_classes):
            inter[2, c] += ((pred_ens == c) & (target == c)).sum()
            union[2, c] += ((pred_ens == c) | (target == c)).sum()

    # Final Stats
    m_ious = []
    
    for i, name in enumerate(["DINOv2", "Segformer", "Ensemble"]):
        valid_indices = union[i] > 0
        ious = inter[i, valid_indices] / union[i, valid_indices]
        m_iou = ious.mean().item()
        m_ious.append(m_iou)
        
        print(f"\n--- {name} Results (mIoU: {m_iou:.4f}) ---")
        for c in range(n_classes):
            if union[i, c] > 0:
                class_iou = (inter[i, c] / union[i, c]).item()
                print(f"Class {c:2d} ({class_names[c]:10s}): {class_iou:.4f}")
            else:
                print(f"Class {c:2d} ({class_names[c]:10s}): No samples found")

    print(f"\n==========================================")
    print(f"Individual DINOv2 mIoU:   {m_ious[0]:.4f}")
    print(f"Individual Segformer mIoU: {m_ious[1]:.4f}")
    print(f"------------------------------------------")
    print(f"Final Ensemble mIoU:       {m_ious[2]:.4f}")
    print(f"Gain over best single:    {((m_ious[2] / max(m_ious[:2])) - 1) * 100:.2f}%")
    print(f"==========================================")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
