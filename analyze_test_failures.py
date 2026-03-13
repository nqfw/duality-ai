"""
Trial Test Visualization
Visualizes failures on the unseen location to diagnose domain shift.
"""
from visualize_ensemble import VizDataset, SegmentationHead, ensemble_inference, device, colorize
import torch
from transformers import SegformerForSemanticSegmentation
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Generating Test Set Analysis Visuals...")
    w, h = 448, 448
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages')
    dataset = VizDataset(test_dir, w, h)
    output_dir = os.path.join(script_dir, 'failure_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Load Models
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device).eval()
    dino_head = SegmentationHead(768, 10, w // 14, h // 14).to(device).eval()
    dino_head.load_state_dict(torch.load("segmentation_head_fast.pth", map_location=device, weights_only=False))
    
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", num_labels=10, ignore_mismatched_sizes=True).to(device)
    ckpt = torch.load("best_b2.pth", map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    segformer.load_state_dict(state_dict, strict=False)
    segformer.eval()

    print("Saving 5 test location samples...")
    for i in range(5):
        img_norm, mask_gt, img_orig, data_id = dataset[i]
        with torch.no_grad():
            prob = ensemble_inference(img_norm.to(device), dino_backbone, dino_head, segformer)
            pred = torch.argmax(prob, dim=0).cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_orig)
        axes[0].set_title("Test Image (Unseen Loc)")
        axes[1].imshow(colorize(mask_gt.numpy()))
        axes[1].set_title("Ground Truth")
        axes[2].imshow(colorize(pred))
        axes[2].set_title("Prediction Error")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"test_fail_{data_id.split('.')[0]}.png"))
        plt.close()

    print(f"Visuals saved to {output_dir}. Review these to see the 'Domain Shift'!")

if __name__ == "__main__":
    main()
