"""
DualityAI Offroad Segmentation — v2: SegFormer-B2, Target > 0.65 mIoU
============================================================================
Key upgrades over v1 (B0, 0.4991 mIoU):
  • SegFormer-B2  (24M params vs 3.7M) — larger "brain"
  • Dice + CrossEntropy combined loss   — directly optimises IoU
  • Warmup → Cosine LR schedule         — more stable early training
  • Partial backbone unfreezing         — lets the backbone adapt
  • 448x448 resolution                  — more spatial detail
  • Confusion matrix + per-class IoU report saved to 'report/' folder
  • Training curve plot
  • Visual sample grid (GT vs Pred overlay)
============================================================================
CMD:
  python train_v2_segformer_b2.py
"""

import os
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
EPOCHS       = 20
BATCH        = 4        # B2 needs more VRAM — use 4; drop to 2 if OOM
SIZE         = 448      # higher resolution for more detail
LR           = 6e-5
WARMUP_EP    = 2        # epochs to warm up LR before cosine decay
NUM_CLASSES  = 10
UNFREEZE_EP  = 5        # after this epoch, unfreeze last 2 encoder blocks

CLASS_NAMES = [
    "Background", "Sky", "Trail", "Vegetation",
    "Obstacle", "Human", "Log", "Rock", "Water", "Vehicle",
]

# Weights: boost rare classes (Log=6, Rock=7, Human=5)
CLASS_WEIGHTS = torch.tensor([
    1.0, 2.0, 1.0, 0.5,
    1.5, 4.0, 5.0, 3.0, 1.0, 0.5
], dtype=torch.float32)

# Mask value mapping
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
_lut = np.zeros(10001, dtype=np.uint8)
for k, v in value_map.items():
    _lut[k] = v

PALETTE = [
    (0, 0, 0),       # Background  — black
    (135, 206, 235), # Sky         — sky-blue
    (139, 69, 19),   # Trail       — brown
    (34, 139, 34),   # Vegetation  — forest-green
    (255, 165, 0),   # Obstacle    — orange
    (255, 0, 0),     # Human       — red
    (101, 67, 33),   # Log         — dark-brown
    (128, 128, 128), # Rock        — grey
    (0, 0, 255),     # Water       — blue
    (255, 255, 0),   # Vehicle     — yellow
]

# ─────────────────────────────────────────────
# MASK UTILS
# ─────────────────────────────────────────────
def convert_mask(path):
    m = np.array(Image.open(path))
    return _lut[m]

def mask_to_color(mask):
    """uint8 class mask → RGB image for visualisation."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, colour in enumerate(PALETTE):
        rgb[mask == cls] = colour
    return rgb

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def train_aug(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def val_aug(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class OffRoadDataset(Dataset):
    def __init__(self, root, transform):
        self.img_dir  = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")
        self.ids      = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(".png"))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img  = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir,  name)), cv2.COLOR_BGR2RGB)
        mask = convert_mask(os.path.join(self.mask_dir, name))
        aug  = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].long(), name

# ─────────────────────────────────────────────
# LOSS: Dice + Cross-Entropy
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        n_cls = logits.shape[1]
        preds = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets, n_cls).permute(0, 3, 1, 2).float()
        inter = (preds * tgt_oh).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + tgt_oh.sum(dim=(2, 3))
        dice  = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self, weights, alpha=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss()
        self.a    = alpha

    def forward(self, logits, targets):
        return self.a * self.ce(logits, targets) + (1 - self.a) * self.dice(logits, targets)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
class RunningMetrics:
    """Accumulates confusion matrix across all batches."""
    def __init__(self, n_cls):
        self.n   = n_cls
        self.mat = np.zeros((n_cls, n_cls), dtype=np.int64)

    def update(self, preds, labels):
        preds  = preds.cpu().numpy().ravel()
        labels = labels.cpu().numpy().ravel()
        mask   = (labels >= 0) & (labels < self.n)
        self.mat += np.bincount(
            self.n * labels[mask].astype(np.int64) + preds[mask],
            minlength=self.n ** 2
        ).reshape(self.n, self.n)

    def class_iou(self):
        diag = np.diag(self.mat)
        denom = self.mat.sum(1) + self.mat.sum(0) - diag
        return np.where(denom > 0, diag / denom, np.nan)

    def miou(self):
        return float(np.nanmean(self.class_iou()))

    def confusion(self):
        return self.mat

# ─────────────────────────────────────────────
# REPORT HELPERS
# ─────────────────────────────────────────────
def save_confusion_matrix(cm, class_names, path):
    # Normalise row-wise for readability
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cmn = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cmn, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Row-normalised frequency")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Normalised Confusion Matrix (Val Set)", fontsize=13, fontweight="bold")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{cmn[i, j]:.2f}" if cmn[i, j] > 0.005 else ""
            colour = "white" if cmn[i, j] > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=colour)

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {path}")

def save_training_curve(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["loss"], marker="o", color="#E74C3C", label="Train Loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["val_miou"], marker="o", color="#2ECC71", label="Val mIoU")
    axes[1].axhline(0.65, linestyle="--", color="#999", label="Target 0.65")
    axes[1].set_title("Validation mIoU", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle("SegFormer-B2 Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Training curve  → {path}")

def save_per_class_iou(class_names, ious, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#2ECC71" if v > 0.65 else "#F39C12" if v > 0.4 else "#E74C3C"
               for v in ious]
    bars = ax.barh(class_names, ious, color=colours, edgecolor="white")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.65, linestyle="--", color="#999", label="Target 0.65")
    for bar, val in zip(bars, ious):
        label = f"{val:.3f}" if not np.isnan(val) else "N/A"
        ax.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)
    ax.set_xlabel("IoU", fontsize=11)
    ax.set_title("Per-Class IoU (Val Set)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Per-class IoU   → {path}")

def save_sample_visual(images, gt_masks, pred_masks, path, n=4):
    """Saves a grid: [Image | GT | Prediction] for n samples."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = axes[None, :]   # guarantee 2-D indexing

    for i in range(n):
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        # de-normalise
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        gt_col   = mask_to_color(gt_masks[i].cpu().numpy()).astype(float) / 255
        pred_col = mask_to_color(pred_masks[i].cpu().numpy()).astype(float) / 255

        axes[i, 0].imshow(img_np);    axes[i, 0].set_title("Input Image",  fontsize=9)
        axes[i, 1].imshow(gt_col);    axes[i, 1].set_title("GT Mask",      fontsize=9)
        axes[i, 2].imshow(pred_col);  axes[i, 2].set_title("Prediction",   fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=np.array(c) / 255, label=n)
               for c, n in zip(PALETTE, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("SegFormer-B2 Sample Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Visual samples  → {path}")

def write_text_report(class_names, ious, history, best_epoch, path):
    miou = float(np.nanmean(ious))
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(" DualityAI Offroad Segmentation -- v2 Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Architecture : SegFormer-B2\n")
        f.write(f"Resolution   : {SIZE}x{SIZE}\n")
        f.write(f"Epochs run   : {EPOCHS}\n")
        f.write(f"Best Epoch   : {best_epoch}\n")
        f.write(f"Best Val mIoU: {max(history['val_miou']):.4f}  (Target > 0.65)\n\n")
        f.write("-" * 40 + "\n")
        f.write("Per-Class IoU (Best Epoch)\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, ious):
            bar = "#" * int(iou * 30) if not np.isnan(iou) else ""
            val = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
            f.write(f"  {name:<12} {val}  {bar}\n")
        f.write(f"\n  Mean IoU     {miou:.4f}\n")
    print(f"  Text report     -> {path}")

# ─────────────────────────────────────────────
# LR SCHEDULER: Warmup → Cosine
# ─────────────────────────────────────────────
def get_lr_lambda(warmup_ep, total_ep):
    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep          # linear warmup
        t = ep - warmup_ep
        T = total_ep - warmup_ep
        return 0.5 * (1 + math.cos(math.pi * t / T))  # cosine decay
    return lr_lambda

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f" SegFormer-B2 | device={device} | target mIoU > 0.65")
    print(f"{'='*60}")

    # Paths
    base    = os.path.dirname(os.path.abspath(__file__))
    data    = os.path.join(base, "Offroad_Segmentation_Training_Dataset")
    test_d  = os.path.join(base, "Offroad_Segmentation_testImages")
    ckpt_d  = os.path.join(base, "checkpoints_b2");  os.makedirs(ckpt_d,  exist_ok=True)
    rep_d   = os.path.join(base, "report");           os.makedirs(rep_d,   exist_ok=True)
    pred_d  = os.path.join(base, "test_predictions"); os.makedirs(pred_d,  exist_ok=True)

    # Data
    train_ds = OffRoadDataset(os.path.join(data, "train"), train_aug(SIZE))
    val_ds   = OffRoadDataset(os.path.join(data, "val"),   val_aug(SIZE))

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    # Model
    print("Loading SegFormer-B2 from nvidia/mit-b2 ...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Freeze backbone initially
    for param in model.segformer.parameters():
        param.requires_grad = False

    criterion = ComboLoss(CLASS_WEIGHTS.to(device), alpha=0.5)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(WARMUP_EP, EPOCHS))
    scaler    = torch.amp.GradScaler("cuda")

    history    = {"loss": [], "val_miou": []}
    best_iou   = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(EPOCHS):

        # Gradual backbone unfreezing
        if epoch == UNFREEZE_EP:
            print(f"\n  [Epoch {epoch+1}] Unfreezing last 2 encoder blocks ...")
            blocks = list(model.segformer.encoder.block.children())
            for blk in blocks[-2:]:
                for p in blk.parameters():
                    p.requires_grad = True
            # Rebuild optimizer with new params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=LR * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                            get_lr_lambda(0, EPOCHS - UNFREEZE_EP))

        # ── Train ──
        model.train()
        ep_loss = 0.0
        pbar = tqdm(train_ld, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast("cuda"):
                out    = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
                loss   = criterion(logits, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = ep_loss / len(train_ld)

        # ── Validate ──
        model.eval()
        metrics   = RunningMetrics(NUM_CLASSES)
        sample_imgs, sample_gt, sample_pred = [], [], []

        with torch.no_grad():
            for imgs, masks, _ in val_ld:
                imgs, masks = imgs.to(device), masks.to(device)
                out    = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
                preds  = torch.argmax(logits, dim=1)
                metrics.update(preds, masks)
                if len(sample_imgs) < 4:
                    sample_imgs.extend(list(imgs.cpu()[:4]))
                    sample_gt.extend(list(masks.cpu()[:4]))
                    sample_pred.extend(list(preds.cpu()[:4]))

        val_miou = metrics.miou()
        cls_iou  = metrics.class_iou()
        history["loss"].append(avg_loss)
        history["val_miou"].append(val_miou)

        print(f"\n  [Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        ckpt = os.path.join(ckpt_d, f"ep{epoch+1:02d}_iou{val_miou:.4f}.pth")
        torch.save(model.state_dict(), ckpt)

        if val_miou > best_iou:
            best_iou   = val_miou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(base, "best_b2.pth"))
            best_cls_iou   = cls_iou
            best_cm        = metrics.confusion()
            best_s_imgs    = sample_imgs[:4]
            best_s_gt      = sample_gt[:4]
            best_s_pred    = sample_pred[:4]
            print(f"  *** NEW BEST: {val_miou:.4f} — saved to best_b2.pth")

    # ── Reports ──
    print(f"\n{'='*60}")
    print(" Generating reports ...")
    save_confusion_matrix(best_cm, CLASS_NAMES, os.path.join(rep_d, "confusion_matrix.png"))
    save_training_curve(history,                os.path.join(rep_d, "training_curve.png"))
    save_per_class_iou(CLASS_NAMES, best_cls_iou, os.path.join(rep_d, "per_class_iou.png"))
    save_sample_visual(best_s_imgs, best_s_gt, best_s_pred, os.path.join(rep_d, "sample_predictions.png"))
    write_text_report(CLASS_NAMES, best_cls_iou, history, best_epoch, os.path.join(rep_d, "summary.txt"))

    # ── Test Inference ──
    print(f"\nRunning inference on test images ...")
    model.load_state_dict(torch.load(os.path.join(base, "best_b2.pth"), map_location=device))
    model.eval()
    test_aug = val_aug(SIZE)
    for fname in tqdm(sorted(os.listdir(test_d)), desc="Inferring"):
        if not fname.lower().endswith((".png", ".jpg")): continue
        raw   = cv2.cvtColor(cv2.imread(os.path.join(test_d, fname)), cv2.COLOR_BGR2RGB)
        oh, ow = raw.shape[:2]
        aug   = test_aug(image=raw)
        inp   = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out   = model(pixel_values=inp)
            logits = F.interpolate(out.logits, size=(oh, ow), mode="bilinear", align_corners=False)
            pred  = torch.argmax(logits, 1).squeeze().cpu().numpy().astype(np.uint8)
        out_name = fname.rsplit(".", 1)[0] + ".png"
        cv2.imwrite(os.path.join(pred_d, out_name), pred)

    print(f"\n{'='*60}")
    print(f" DONE. Best Val mIoU={best_iou:.4f} (epoch {best_epoch})")
    print(f" Reports   → {rep_d}/")
    print(f" Weights   → best_b2.pth")
    print(f" Predictions → {pred_d}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
