"""
FastAPI Inference Server for Desert Pathfinder AI
Exposes: POST /predict  — accepts an image, returns colorized segmentation mask as PNG
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import io
import os
import time

# ── Config ──────────────────────────────────────────────────────
W, H = 784, 448
N_CLASSES = 10
CLASS_NAMES = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
               "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
PALETTE = [
    [34, 139, 34],   # Trees - Forest Green
    [0, 200, 80],    # Lush Bushes - Bright Green
    [210, 180, 140], # Dry Grass - Tan
    [139, 90, 43],   # Dry Bushes - Brown
    [105, 105, 105], # Ground Clutter - Gray
    [255, 20, 147],  # Flowers - Deep Pink
    [101, 67, 33],   # Logs - Dark Brown
    [128, 128, 128], # Rocks - Gray
    [194, 178, 128], # Landscape - Sand
    [87, 181, 231],  # Sky - Blue
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ── Model Architecture ───────────────────────────────────────────
class SegmentationHead(nn.Module):
    def __init__(self, in_chan, out_chan, size_w, size_h):
        super().__init__()
        self.sz_w = size_w // 14
        self.sz_h = size_h // 14
        self.stem = nn.Sequential(nn.Conv2d(in_chan, 256, 7, padding=3), nn.BatchNorm2d(256), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 1), nn.GELU()
        )
        self.classifier = nn.Conv2d(128, out_chan, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.sz_h, self.sz_w, C).permute(0, 3, 1, 2)
        return self.classifier(self.block2(self.block1(self.stem(x))))

# ── Load Models at startup ───────────────────────────────────────
print(f"Loading models on {DEVICE}...")
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(DEVICE).eval()
dino_head = SegmentationHead(768, N_CLASSES, W, H).to(DEVICE).eval()

# Prioritize Turbo weights if available, otherwise fallback to HailMary or Fast
head_path = "segmentation_head_turbo.pth" if os.path.exists("segmentation_head_turbo.pth") \
    else "segmentation_head_hailmary.pth" if os.path.exists("segmentation_head_hailmary.pth") \
    else "segmentation_head_fast.pth"
    
backbone_path = "dino_backbone_turbo.pth" if os.path.exists("dino_backbone_turbo.pth") \
    else "dino_backbone_hailmary.pth" if os.path.exists("dino_backbone_hailmary.pth") \
    else None

print(f"  Using head weights: {head_path}")
dino_head.load_state_dict(torch.load(head_path, map_location=DEVICE, weights_only=False))

if backbone_path:
    print(f"  Using backbone weights: {backbone_path}")
    dino_backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE, weights_only=False))

segformer = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    num_labels=N_CLASSES, ignore_mismatched_sizes=True
).to(DEVICE).eval()
ckpt_path = "best_b2.pth" if os.path.exists("best_b2.pth") else "best_segformer.pth"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    segformer.load_state_dict(state_dict, strict=False)
print("Models loaded! Server ready.")

# ── Helper Functions ─────────────────────────────────────────────
def preprocess(img_pil: Image.Image):
    img = img_pil.convert("RGB").resize((W, H), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    return NORMALIZE(tensor).to(DEVICE)

def ensemble_predict(img_tensor):
    """Run weighted ensemble with horizontal flip TTA."""
    imgs = torch.stack([img_tensor, TF.hflip(img_tensor)])
    with torch.no_grad():
        # DINOv2
        feats = dino_backbone.forward_features(imgs)["x_norm_patchtokens"]
        dino_logits = dino_head(feats)
        dino_logits = F.interpolate(dino_logits, size=(H, W), mode='bilinear', align_corners=False)

        # Segformer
        seg_out = segformer(pixel_values=imgs).logits
        seg_logits = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)

    # Flip TTA back
    dino_prob = torch.softmax(dino_logits, dim=1)
    seg_prob = torch.softmax(seg_logits, dim=1)
    dino_prob[1] = TF.hflip(dino_prob[1])
    seg_prob[1] = TF.hflip(seg_prob[1])

    dino_avg = dino_prob.mean(0, keepdim=True)
    seg_avg = seg_prob.mean(0, keepdim=True)

    # Boost DINO to 75% as requested, Segformer 25%
    blended = 0.75 * dino_avg + 0.25 * seg_avg
    return blended.squeeze(0)

def colorize(pred_mask: np.ndarray) -> Image.Image:
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        rgb[pred_mask == i] = color
    return Image.fromarray(rgb)

def get_class_percentages(pred_mask: np.ndarray) -> dict:
    total = pred_mask.size
    result = {}
    for i, name in enumerate(CLASS_NAMES):
        pct = round((pred_mask == i).sum() / total * 100, 2)
        if pct > 0:
            result[name] = pct
    return result

# ── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(title="Desert Pathfinder AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Desert Pathfinder AI is running", "device": str(DEVICE)}

@app.get("/health")
def health():
    return {"status": "ok", "model": head_path, "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents))
    img_tensor = preprocess(img_pil)
    prob = ensemble_predict(img_tensor)
    pred = prob.argmax(0).cpu().numpy()
    colored = colorize(pred)
    distribution = get_class_percentages(pred)
    latency_ms = round((time.time() - start) * 1000, 1)
    buf = io.BytesIO()
    colored.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Latency-Ms": str(latency_ms),
            "X-Class-Distribution": str(distribution),
            "Access-Control-Expose-Headers": "X-Latency-Ms, X-Class-Distribution"
        }
    )

@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    import base64
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents))
    img_tensor = preprocess(img_pil)
    prob = ensemble_predict(img_tensor)
    pred = prob.argmax(0).cpu().numpy()
    colored = colorize(pred)
    distribution = get_class_percentages(pred)
    buf = io.BytesIO()
    colored.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return JSONResponse({
        "mask_base64": b64,
        "class_distribution": distribution,
        "model": head_path,
        "resolution": f"{W}x{H}"
    })
