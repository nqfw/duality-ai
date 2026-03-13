"""
Official Test Set Evaluation Script
Evaluates the ensemble on the unseen 'testImages' location.
"""
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

# Reuse architecture and logic
from merge_models import SegmentationHead, EvalDataset, ensemble_inference, device, n_classes, class_names
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

def main():
    print("Initializing Official Test Set Evaluation (Unseen Location)...")
    # Higher resolution for final test run
    w, h = 784, 448 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages')
    test_loader = DataLoader(EvalDataset(test_dir, w, h), batch_size=1, shuffle=False)

    # Load Models
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device).eval()
    # Load robust backbone weights
    dino_backbone.load_state_dict(torch.load("dino_backbone_robust.pth", map_location=device, weights_only=False))
    
    dino_head = SegmentationHead(768, n_classes, w // 14, h // 14).to(device).eval()
    dino_head.load_state_dict(torch.load("segmentation_head_robust.pth", map_location=device, weights_only=False))
    
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", num_labels=n_classes, ignore_mismatched_sizes=True).to(device)
    ckpt = torch.load("best_b2.pth", map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not any(k.startswith("model.") for k in segformer.state_dict().keys()) and any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    segformer.load_state_dict(state_dict, strict=False)
    segformer.eval()

    # Reset metrics for test set
    test_inter = torch.zeros(n_classes, device=device)
    test_union = torch.zeros(n_classes, device=device)

    print(f"\nRunning hidden test set inference on {len(test_loader)} images...")
    for img_tensor, mask in tqdm(test_loader):
        img_tensor, mask = img_tensor.to(device).squeeze(0), mask.to(device)
        target = mask.view(-1)
        
        with torch.no_grad():
            blended_prob = ensemble_inference(img_tensor, dino_backbone, dino_head, segformer)
            pred = torch.argmax(blended_prob, dim=1).view(-1)
            for c in range(n_classes):
                test_inter[c] += ((pred == c) & (target == c)).sum()
                test_union[c] += ((pred == c) | (target == c)).sum()

    valid_indices = test_union > 0
    test_ious = test_inter[valid_indices] / test_union[valid_indices]
    final_test_miou = test_ious.mean().item()

    print(f"\n==========================================")
    print(f"OFFICIAL TEST SET mIoU: {final_test_miou:.4f}")
    print(f"==========================================")
    for c in range(n_classes):
        if test_union[c] > 0:
            iou = (test_inter[c] / test_union[c]).item()
            print(f"Class {c:2d} ({class_names[c]:12s}): {iou:.4f}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
