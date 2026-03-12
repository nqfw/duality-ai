"""
Standalone report generator — uses best_b2.pth (no retraining needed)
Run: python generate_report.py
"""
import os, math, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings; warnings.filterwarnings("ignore")

# ── Config (must match training) ──────────────────────────────
NUM_CLASSES = 10
SIZE        = 448
BATCH       = 4
CLASS_NAMES = ["Background","Sky","Trail","Vegetation",
               "Obstacle","Human","Log","Rock","Water","Vehicle"]
PALETTE     = [
    (0,0,0),(135,206,235),(139,69,19),(34,139,34),(255,165,0),
    (255,0,0),(101,67,33),(128,128,128),(0,0,255),(255,255,0)
]
value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
_lut = np.zeros(10001, dtype=np.uint8)
for k,v in value_map.items(): _lut[k]=v

# Training history from the run
HISTORY = {
    "loss": [0.9047,0.7395,0.7007,0.6821,0.6679,
             0.6197,0.5961,0.5673,0.5496,0.5347,
             0.5235,0.5163,0.5074,0.5074,0.5051,
             0.5008,0.4985,0.4950,0.4956,0.4956],
    "val_miou": [0.4686,0.5098,0.5273,0.5415,0.5504,
                 0.5640,0.5662,0.5681,0.5697,0.5706,
                 0.5739,0.5789,0.5787,0.5788,0.5793,
                 0.5810,0.5801,0.5821,0.5800,0.5799],
}
BEST_EPOCH = 18

# ── Helpers ───────────────────────────────────────────────────
def convert_mask(path):
    return _lut[np.array(Image.open(path))]

def mask_to_color(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, col in enumerate(PALETTE): rgb[mask==cls] = col
    return rgb

def val_aug():
    return A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

class ValDataset(Dataset):
    def __init__(self, root, transform):
        self.img_dir  = os.path.join(root, "Color_Images")
        self.mask_dir = os.path.join(root, "Segmentation")
        self.ids = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith(".png"))
        self.tf = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        n   = self.ids[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir,  n)), cv2.COLOR_BGR2RGB)
        msk = convert_mask(os.path.join(self.mask_dir, n))
        aug = self.tf(image=img, mask=msk)
        return aug["image"], aug["mask"].long()

class RunningMetrics:
    def __init__(self, n):
        self.n   = n
        self.mat = np.zeros((n,n), dtype=np.int64)
    def update(self, preds, labels):
        p = preds.cpu().numpy().ravel()
        l = labels.cpu().numpy().ravel()
        m = (l>=0)&(l<self.n)
        self.mat += np.bincount(self.n*l[m].astype(np.int64)+p[m], minlength=self.n**2).reshape(self.n,self.n)
    def class_iou(self):
        d = np.diag(self.mat); dm = self.mat.sum(1)+self.mat.sum(0)-d
        return np.where(dm>0, d/dm, np.nan)
    def miou(self): return float(np.nanmean(self.class_iou()))
    def confusion(self): return self.mat

# ── Report writers ────────────────────────────────────────────
def save_confusion_matrix(cm, names, path):
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cmn = cm.astype(float)/row_sums
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Normalised Confusion Matrix (Val Set)", fontweight="bold")
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{cmn[i,j]:.2f}" if cmn[i,j]>0.005 else ""
            ax.text(j,i,txt,ha="center",va="center",fontsize=7,
                    color="white" if cmn[i,j]>0.6 else "black")
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix -> {path}")

def save_training_curve(history, path):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(history["loss"], marker="o", color="#E74C3C", label="Train Loss")
    axes[0].set_title("Training Loss", fontweight="bold"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(history["val_miou"], marker="o", color="#2ECC71", label="Val mIoU")
    axes[1].axhline(0.65, linestyle="--", color="#999", label="Target 0.65")
    axes[1].set_title("Validation mIoU", fontweight="bold"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3); axes[1].legend()
    plt.suptitle("SegFormer-B2 Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  Training curve  -> {path}")

def save_per_class_iou(names, ious, path):
    fig, ax = plt.subplots(figsize=(10,6))
    colours = ["#2ECC71" if v>0.65 else "#F39C12" if v>0.4 else "#E74C3C" for v in ious]
    bars = ax.barh(names, ious, color=colours, edgecolor="white")
    ax.set_xlim(0,1.0); ax.axvline(0.65, linestyle="--", color="#999", label="Target 0.65")
    for bar, val in zip(bars, ious):
        lbl = f"{val:.3f}" if not np.isnan(val) else "N/A"
        ax.text(min(val+0.02, 0.95), bar.get_y()+bar.get_height()/2, lbl, va="center", fontsize=9)
    ax.set_xlabel("IoU"); ax.set_title("Per-Class IoU (Val Set)", fontweight="bold"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  Per-class IoU   -> {path}")

def save_sample_visual(images, gt_masks, pred_masks, path, n=4):
    n = min(n, len(images))
    fig, axes = plt.subplots(n,3,figsize=(12, n*4))
    if n==1: axes = axes[None,:]
    for i in range(n):
        img_np = images[i].cpu().permute(1,2,0).numpy()
        img_np = np.clip(img_np*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]),0,1)
        axes[i,0].imshow(img_np);                              axes[i,0].set_title("Input"); axes[i,0].axis("off")
        axes[i,1].imshow(mask_to_color(gt_masks[i].cpu().numpy()).astype(float)/255);   axes[i,1].set_title("GT"); axes[i,1].axis("off")
        axes[i,2].imshow(mask_to_color(pred_masks[i].cpu().numpy()).astype(float)/255); axes[i,2].set_title("Pred"); axes[i,2].axis("off")
    patches = [mpatches.Patch(color=np.array(c)/255, label=nm) for c,nm in zip(PALETTE,CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5,-0.02))
    plt.suptitle("SegFormer-B2 Sample Predictions", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Samples         -> {path}")

def write_text_report(names, ious, history, best_epoch, path):
    miou = float(np.nanmean(ious))
    with open(path, "w", encoding="utf-8") as f:
        f.write("="*60+"\n")
        f.write(" DualityAI Offroad Segmentation -- v2 Report\n")
        f.write("="*60+"\n\n")
        f.write(f"Architecture : SegFormer-B2\n")
        f.write(f"Resolution   : {SIZE}x{SIZE}\n")
        f.write(f"Epochs run   : {len(history['loss'])}\n")
        f.write(f"Best Epoch   : {best_epoch}\n")
        f.write(f"Best Val mIoU: {max(history['val_miou']):.4f}  (Target > 0.65)\n\n")
        f.write("-"*40+"\n")
        f.write("Per-Class IoU (Best Epoch)\n")
        f.write("-"*40+"\n")
        for name, iou in zip(names, ious):
            bar = "#" * int(iou*30) if not np.isnan(iou) else ""
            val = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
            f.write(f"  {name:<12} {val}  {bar}\n")
        f.write(f"\n  Mean IoU     {miou:.4f}\n")
    print(f"  Text report     -> {path}")

# ── Main ──────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base   = os.path.dirname(os.path.abspath(__file__))
    rep_d  = os.path.join(base, "report"); os.makedirs(rep_d, exist_ok=True)

    print("Loading best_b2.pth ...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(device)
    model.load_state_dict(torch.load(os.path.join(base, "best_b2.pth"), map_location=device))
    model.eval()

    print("Running validation to collect metrics ...")
    val_ds = ValDataset(os.path.join(base, "Offroad_Segmentation_Training_Dataset", "val"), val_aug())
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    metrics = RunningMetrics(NUM_CLASSES)
    s_imgs, s_gt, s_pred = [], [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_ld, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)
            out    = model(pixel_values=imgs)
            logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds  = torch.argmax(logits, dim=1)
            metrics.update(preds, masks)
            if len(s_imgs) < 4:
                s_imgs.extend(list(imgs.cpu()[:4]));  s_gt.extend(list(masks.cpu()[:4]));  s_pred.extend(list(preds.cpu()[:4]))

    cls_iou = metrics.class_iou()
    print(f"\nVal mIoU: {metrics.miou():.4f}\n")

    print("Generating reports ...")
    save_confusion_matrix(metrics.confusion(), CLASS_NAMES, os.path.join(rep_d, "confusion_matrix.png"))
    save_training_curve(HISTORY, os.path.join(rep_d, "training_curve.png"))
    save_per_class_iou(CLASS_NAMES, cls_iou, os.path.join(rep_d, "per_class_iou.png"))
    save_sample_visual(s_imgs[:4], s_gt[:4], s_pred[:4], os.path.join(rep_d, "sample_predictions.png"))
    write_text_report(CLASS_NAMES, cls_iou, HISTORY, BEST_EPOCH, os.path.join(rep_d, "summary.txt"))
    print(f"\nDone! All reports saved to {rep_d}/")

if __name__ == "__main__":
    main()
