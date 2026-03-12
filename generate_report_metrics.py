"""
Final Evaluation & Report Generator
Outputs: 
1. Global mIoU
2. Confusion Matrix (CSV + PNG)
3. Inference Speed Benchmark
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Reuse architecture and logic from merge_models
from merge_models import SegmentationHead, EvalDataset, ensemble_inference, device, n_classes, class_names, value_map
from transformers import SegformerForSemanticSegmentation

def main():
    print("Starting Final Evaluation for Submission Report...")
    # Optimized resolution for speed-balance: 336x196 (multiples of 14)
    w, h = 336, 196 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')
    val_loader = DataLoader(EvalDataset(val_dir, w, h), batch_size=1, shuffle=False)

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

    all_preds = []
    all_targets = []
    latencies = []

    print(f"\nBenchmarking {len(val_loader)} images...")
    for img_tensor, mask in tqdm(val_loader):
        img_tensor = img_tensor.to(device).squeeze(0)
        mask = mask.to(device).view(-1).cpu().numpy()
        
        # Latency measurement
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            blended_prob = ensemble_inference(img_tensor, dino_backbone, dino_head, segformer)
            pred = torch.argmax(blended_prob, dim=1).view(-1).cpu().numpy()
        
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000) # ms

        # Collect for confusion matrix (sampled to avoid memory issues)
        # We sample 1% of pixels for the matrix
        sample_indices = np.random.choice(len(mask), len(mask)//100, replace=False)
        all_preds.extend(pred[sample_indices])
        all_targets.extend(mask[sample_indices])

    # 1. Inference Speed
    avg_latency = np.mean(latencies)
    print(f"\n==========================================")
    print(f"AVG Inference Speed: {avg_latency:.2f} ms")
    if avg_latency < 50:
        print("SUCCESS: SPEED GOAL REACHED (<50ms)")
    else:
        print("WARNING: SPEED GOAL EXCEEDED (>50ms). Consider reducing resolution.")

    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(n_classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("SUCCESS: Confusion Matrix saved to confusion_matrix.png")

    # 3. CSV Report
    report_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    report_df.to_csv('performance_report.csv')
    print("SUCCESS: Detailed stats saved to performance_report.csv")
    print(f"==========================================")

if __name__ == "__main__":
    main()
