"""
Ensemble Visualization Tool
Generates side-by-side comparisons [Original | Ground Truth | Prediction]
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
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Configuration & Color Mapping
# ==========================================
# Define a high-contrast color palette for 10 classes
palette = [
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: Road (Red)
    [0, 255, 0],     # 2: Trail (Green)
    [0, 0, 255],     # 3: Grass (Blue)
    [255, 255, 0],   # 4: Vegetation (Yellow)
    [0, 255, 255],   # 5: Sky (Cyan)
    [255, 0, 255],   # 6: Obstacle (Magenta)
    [192, 192, 192], # 7: Misc (Silver)
    [128, 0, 0],     # 8: Unknown (Maroon)
    [128, 128, 0],   # 9: Boundary (Olive)
]

def colorize(mask):
    """Converts a label mask [H, W] to an RGB image [H, W, 3]"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        rgb[mask == i] = color
    return rgb

# Architecture & Dataset (Keep consistent with merge_models.py)
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tw, th):
        super().__init__()
        self.tw, self.th = tw, th
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.BatchNorm2d(256), nn.GELU())
        self.block1 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 256, 1), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.th, self.tw, C).permute(0, 3, 1, 2)
        return self.classifier(self.block2(self.block1(self.stem(x))))

value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class VizDataset(Dataset):
    def __init__(self, data_dir, w=784, h=448):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        # Sort to get deterministic images
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.w, self.h = w, h
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self): return min(len(self.data_ids), 20) # Only viz first 20

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image_pil = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB").resize((self.w, self.h), Image.BILINEAR)
        mask_pil = Image.open(os.path.join(self.masks_dir, data_id))
        mask_pil = convert_mask(mask_pil).resize((self.w, self.h), Image.NEAREST)
        image_tensor = TF.to_tensor(image_pil)
        mask_array = np.array(mask_pil)
        return self.normalize(image_tensor), torch.from_numpy(mask_array).long(), np.array(image_pil), data_id

# ==========================================
# 2. Ensemble Inference (Consistent Logic)
# ==========================================
def ensemble_inference(image, dino_backbone, dino_head, segformer, weights=[0.4, 0.6]):
    batch_imgs = torch.stack([image, TF.hflip(image)]) 
    with torch.no_grad():
        feats = dino_backbone.forward_features(batch_imgs)["x_norm_patchtokens"]
        logits_dino = dino_head(feats)
        h, w = image.shape[-2], image.shape[-1]
        logits_dino = F.interpolate(logits_dino, size=(h, w), mode='bilinear', align_corners=False)
        outputs = segformer(batch_imgs)
        logits_seg = F.interpolate(outputs.logits, size=(h, w), mode='bilinear', align_corners=False)

    logits_dino[1] = TF.hflip(logits_dino[1])
    logits_seg[1] = TF.hflip(logits_seg[1])
    prob_dino = F.softmax(logits_dino, dim=1)
    prob_seg = F.softmax(logits_seg, dim=1)
    final_prob = (weights[0] * (prob_dino[0] + prob_dino[1])/2.0) + (weights[1] * (prob_seg[0] + prob_seg[1])/2.0)
    return final_prob

def main():
    print("Initializing Visualization...")
    w, h = 784, 448
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')
    dataset = VizDataset(val_dir, w, h)
    output_dir = os.path.join(script_dir, 'visualization_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load Models
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device).eval()
    dino_head = SegmentationHead(768, len(value_map), w // 14, h // 14).to(device).eval()
    dino_head.load_state_dict(torch.load("segmentation_head_fast.pth", map_location=device, weights_only=False))
    
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", num_labels=len(value_map), ignore_mismatched_sizes=True).to(device)
    ckpt = torch.load("best_b2.pth", map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not any(k.startswith("model.") for k in segformer.state_dict().keys()) and any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    segformer.load_state_dict(state_dict, strict=False)
    segformer.eval()

    num_samples = 10
    print(f"Generating {num_samples} comparisons in {output_dir}...")

    for i in range(num_samples):
        img_norm, mask_gt, img_orig, data_id = dataset[i]
        img_norm = img_norm.to(device)
        
        # Run Inference
        prob = ensemble_inference(img_norm, dino_backbone, dino_head, segformer)
        pred = torch.argmax(prob, dim=0).cpu().numpy()
        
        # Create Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_orig)
        axes[0].set_title(f"Image: {data_id}")
        axes[0].axis('off')
        
        axes[1].imshow(colorize(mask_gt.numpy()))
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(colorize(pred))
        axes[2].set_title("Ensemble Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"compare_{data_id.split('.')[0]}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

    print("\nVisualization complete! Check the 'visualization_results' folder.")

if __name__ == "__main__":
    main()
