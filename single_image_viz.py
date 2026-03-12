import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import os

# Reuse logic from merge_models
from merge_models import SegmentationHead, ensemble_inference, device, n_classes, value_map

# Define palette and colorize locally
palette = [
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [0, 255, 255], [255, 0, 255], [192, 192, 192], [128, 0, 0], [128, 128, 0],
]

def colorize(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        rgb[mask == i] = color
    return rgb

def main():
    img_path = r"C:\Users\dipak\OneDrive\Desktop\New folder (3)\duality-ai\Offroad_Segmentation_Training_Dataset\val\Color_Images\ww10000528.png"
    w, h = 784, 448
    
    print(f"Loading models and testing image: {os.path.basename(img_path)}")
    
    # Load Models
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device).eval()
    dino_head = SegmentationHead(768, n_classes, w // 14, h // 14).to(device).eval()
    dino_head.load_state_dict(torch.load("segmentation_head_fast.pth", map_location=device, weights_only=False))
    
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", num_labels=n_classes, ignore_mismatched_sizes=True).to(device)
    ckpt = torch.load("best_b2.pth", map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not any(k.startswith("model.") for k in segformer.state_dict().keys()) and any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    segformer.load_state_dict(state_dict, strict=False)
    segformer.eval()

    # Preprocess Image
    img_pil = Image.open(img_path).convert("RGB").resize((w, h), Image.BILINEAR)
    img_tensor = TF.to_tensor(img_pil)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_norm = normalize(img_tensor).to(device)

    # Inference
    with torch.no_grad():
        prob = ensemble_inference(img_norm, dino_backbone, dino_head, segformer)
        pred = torch.argmax(prob, dim=0).cpu().numpy()

    # Display Result
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(colorize(pred))
    axes[1].set_title("Ensemble Prediction")
    axes[1].axis('off')
    
    print("Opening pop-up window...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
