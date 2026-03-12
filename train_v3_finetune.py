"""
DualityAI v3 — Fast Fine-tune from best_b2.pth → target > 0.60 mIoU
======================================================================
Key ideas:
  • Resumes from best_b2.pth (skips 20 wasted epochs)
  • Unfreezes FULL backbone with very small LR (differential LR)
  • 7 epochs (~30 mins on 6GB).  Stop when IoU > 0.60.
  • Saves updated report to report/
======================================================================
CMD:
  python train_v3_finetune.py
"""

import os, cv2, math, numpy as np, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
EPOCHS        = 7          # 7 × ~4.5 min ≈ 30 min
BATCH         = 4
SIZE          = 448        # same as v2
NUM_CLASSES   = 10
LR_HEAD       = 2e-5      # head — can learn faster
LR_BACKBONE   = 3e-6      # backbone — careful not to destroy it
TARGET_IOU    = 0.60      # stop early if we hit it

CLASS_NAMES = ["Background","Sky","Trail","Vegetation",
               "Obstacle","Human","Log","Rock","Water","Vehicle"]

CLASS_WEIGHTS = torch.tensor([
    1.0, 2.0, 1.0, 0.5, 1.5, 4.0, 5.0, 3.0, 1.0, 0.5
], dtype=torch.float32)

value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
_lut = np.zeros(10001, dtype=np.uint8)
for k,v in value_map.items(): _lut[k]=v

PALETTE = [(0,0,0),(135,206,235),(139,69,19),(34,139,34),(255,165,0),
           (255,0,0),(101,67,33),(128,128,128),(0,0,255),(255,255,0)]

# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
def convert_mask(path):
    return _lut[np.array(Image.open(path))]

def mask_to_color(mask):
    rgb = np.zeros((*mask.shape,3), dtype=np.uint8)
    for cls,col in enumerate(PALETTE): rgb[mask==cls]=col
    return rgb

# ──────────────────────────────────────────────────────────────
# DATASET — slightly stronger augmentation this round
# ──────────────────────────────────────────────────────────────
def train_aug():
    return A.Compose([
        A.Resize(SIZE, SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
        A.HueSaturationValue(15, 30, 15, p=0.4),
        A.GaussianBlur(blur_limit=(3,7), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
        A.CoarseDropout(max_holes=6, max_height=40, max_width=40, p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

def val_aug():
    return A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

class OffRoadDataset(Dataset):
    def __init__(self, root, tf):
        self.img_dir  = os.path.join(root,"Color_Images")
        self.mask_dir = os.path.join(root,"Segmentation")
        self.ids = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(".png"))
        self.tf = tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        n   = self.ids[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir,n)), cv2.COLOR_BGR2RGB)
        msk = convert_mask(os.path.join(self.mask_dir,n))
        aug = self.tf(image=img, mask=msk)
        return aug["image"], aug["mask"].long()

# ──────────────────────────────────────────────────────────────
# LOSS  (same Dice + CE combo that worked in v2)
# ──────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        n = logits.shape[1]
        p = F.softmax(logits,1)
        t = F.one_hot(targets,n).permute(0,3,1,2).float()
        inter = (p*t).sum((2,3)); union = p.sum((2,3))+t.sum((2,3))
        return (1 - (2*inter+1)/(union+1)).mean()

class ComboLoss(nn.Module):
    def __init__(self, w):
        super().__init__(); self.ce=nn.CrossEntropyLoss(weight=w); self.dice=DiceLoss()
    def forward(self, x, y): return 0.5*self.ce(x,y)+0.5*self.dice(x,y)

# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
class RunningMetrics:
    def __init__(self, n): self.n=n; self.mat=np.zeros((n,n),dtype=np.int64)
    def update(self,p,l):
        p=p.cpu().numpy().ravel(); l=l.cpu().numpy().ravel(); m=(l>=0)&(l<self.n)
        self.mat+=np.bincount(self.n*l[m].astype(np.int64)+p[m],minlength=self.n**2).reshape(self.n,self.n)
    def class_iou(self):
        d=np.diag(self.mat); dm=self.mat.sum(1)+self.mat.sum(0)-d
        return np.where(dm>0,d/dm,np.nan)
    def miou(self): return float(np.nanmean(self.class_iou()))

# ──────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────
def write_report(class_names, ious, history, best_epoch, rep_d):
    # Summary txt
    miou = float(np.nanmean(ious))
    with open(os.path.join(rep_d,"summary_v3.txt"), "w", encoding="utf-8") as f:
        f.write("="*60+"\n DualityAI Segmentation -- v3 Fine-tune Report\n"+"="*60+"\n\n")
        f.write(f"Resume from  : best_b2.pth (0.5821 mIoU)\n")
        f.write(f"Architecture : SegFormer-B2 (full backbone unfrozen)\n")
        f.write(f"Resolution   : {SIZE}x{SIZE}\n")
        f.write(f"Fine-tune ep : {len(history['loss'])}\n")
        f.write(f"Best Epoch   : {best_epoch}\n")
        f.write(f"Best Val mIoU: {max(history['val_miou']):.4f}\n\n")
        f.write("-"*40+"\nPer-Class IoU\n"+"-"*40+"\n")
        for name,iou in zip(class_names,ious):
            bar="#"*int(iou*30) if not np.isnan(iou) else ""
            val=f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<12} {val}  {bar}\n")
        f.write(f"\n  Mean IoU     {miou:.4f}\n")

    # Per-class bar chart
    fig, ax = plt.subplots(figsize=(10,6))
    colours=["#2ECC71" if v>0.65 else "#F39C12" if v>0.40 else "#E74C3C" for v in ious]
    bars=ax.barh(class_names,ious,color=colours,edgecolor="white")
    ax.set_xlim(0,1.0); ax.axvline(0.60,linestyle="--",color="#2980B9",label="Target 0.60")
    ax.axvline(0.65,linestyle="--",color="#999",label="Stretch 0.65")
    for bar,val in zip(bars,ious):
        lbl=f"{val:.3f}" if not np.isnan(val) else "N/A"
        ax.text(min(val+0.02,0.95),bar.get_y()+bar.get_height()/2,lbl,va="center",fontsize=9)
    ax.set_xlabel("IoU"); ax.set_title("Per-Class IoU — v3 Fine-tune",fontweight="bold"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(rep_d,"per_class_iou_v3.png"),dpi=160,bbox_inches="tight"); plt.close()

    # Training curve
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(history["loss"],marker="o",color="#E74C3C",label="Loss")
    axes[0].set_title("Loss (fine-tune)",fontweight="bold"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(history["val_miou"],marker="o",color="#2ECC71",label="Val mIoU")
    axes[1].axhline(0.60,linestyle="--",color="#2980B9",label="Target 0.60")
    axes[1].set_title("Val mIoU (fine-tune)",fontweight="bold"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3); axes[1].legend()
    plt.suptitle("SegFormer-B2 v3 Fine-tune",fontsize=14,fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(rep_d,"training_curve_v3.png"),dpi=160,bbox_inches="tight"); plt.close()

    print(f"  Reports saved to {rep_d}/")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base   = os.path.dirname(os.path.abspath(__file__))
    data   = os.path.join(base,"Offroad_Segmentation_Training_Dataset")
    test_d = os.path.join(base,"Offroad_Segmentation_testImages")
    rep_d  = os.path.join(base,"report");    os.makedirs(rep_d, exist_ok=True)
    pred_d = os.path.join(base,"test_predictions"); os.makedirs(pred_d, exist_ok=True)

    print("="*60)
    print(f" v3 Fine-tune  |  Resuming from best_b2.pth  |  target > {TARGET_IOU}")
    print("="*60)

    # Data
    train_ds = OffRoadDataset(os.path.join(data,"train"), train_aug())
    val_ds   = OffRoadDataset(os.path.join(data,"val"),   val_aug())
    train_ld = DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    val_ld   = DataLoader(val_ds,  batch_size=BATCH,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)

    # Load model
    print("Loading SegFormer-B2 + best_b2.pth weights ...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(device)
    state = torch.load(os.path.join(base,"best_b2.pth"), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print("  Weights loaded. Unfreezing FULL backbone with differential LR.")

    # Differential LR groups
    backbone_params = list(model.segformer.parameters())
    head_params     = list(model.decode_head.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda")
    criterion = ComboLoss(CLASS_WEIGHTS.to(device))

    history    = {"loss":[],"val_miou":[]}
    best_iou   = 0.5821   # start from known best
    best_epoch = 0
    best_cls_iou = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        ep_loss = 0.0
        pbar = tqdm(train_ld, desc=f"FineTune {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast("cuda"):
                out    = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss   = criterion(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # Validate
        model.eval()
        metrics = RunningMetrics(NUM_CLASSES)
        with torch.no_grad():
            for imgs, masks in val_ld:
                imgs, masks = imgs.to(device), masks.to(device)
                out    = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                metrics.update(torch.argmax(logits,1), masks)

        val_miou = metrics.miou()
        history["loss"].append(ep_loss/len(train_ld))
        history["val_miou"].append(val_miou)
        print(f"\n  [Epoch {epoch+1}] Loss: {ep_loss/len(train_ld):.4f} | Val mIoU: {val_miou:.4f}")

        torch.save(model.state_dict(), os.path.join(base, f"finetune_ep{epoch+1}_iou{val_miou:.4f}.pth"))

        if val_miou > best_iou:
            best_iou   = val_miou
            best_epoch = epoch+1
            best_cls_iou = metrics.class_iou()
            torch.save(model.state_dict(), os.path.join(base, "best_b2.pth"))  # overwrite with better weights
            print(f"  *** NEW BEST: {val_miou:.4f} — best_b2.pth updated")

    # Report
    if best_cls_iou is not None:
        print("\nGenerating reports ...")
        write_report(CLASS_NAMES, best_cls_iou, history, best_epoch, rep_d)
    else:
        print(f"\nNo improvement beyond 0.5821 — best_b2.pth unchanged.")

    # Test Inference with best weights
    print("\nRunning test inference ...")
    model.load_state_dict(torch.load(os.path.join(base,"best_b2.pth"), map_location=device, weights_only=False), strict=False)
    model.eval()
    tf = val_aug()
    for fname in tqdm(sorted(os.listdir(test_d)), desc="Inferring"):
        if not fname.lower().endswith((".png",".jpg")): continue
        raw = cv2.cvtColor(cv2.imread(os.path.join(test_d,fname)), cv2.COLOR_BGR2RGB)
        oh,ow = raw.shape[:2]
        aug = tf(image=raw)
        with torch.no_grad():
            out = model(pixel_values=aug["image"].unsqueeze(0).to(device))
            logits = F.interpolate(out.logits, size=(oh,ow), mode="bilinear", align_corners=False)
            pred = torch.argmax(logits,1).squeeze().cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(pred_d, fname.rsplit(".",1)[0]+".png"), pred)

    print(f"\n{'='*60}")
    print(f" DONE. Best Val mIoU = {best_iou:.4f}")
    print(f" Weights -> best_b2.pth | Predictions -> {pred_d}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
