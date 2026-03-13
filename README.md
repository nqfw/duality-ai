# 🌵 Desert Atlas AI: Offroad Terrain Segmentation

**Validation mIoU: 61.42%** | **Official Test mIoU: 25.36%**

[![Report](https://img.shields.io/badge/Report-Project%20Analysis-red.svg)](https://docs.google.com/document/d/1M6RaimGfEBxcZ00kvSOTMjiEldBgg0y7/edit)
[![mIoU](https://img.shields.io/badge/mIoU-61.42%25-cyan.svg)](https://github.com/nqfw/duality-ai)
[![Test-mIoU](https://img.shields.io/badge/Test--mIoU-25.36%25-orange.svg)](https://github.com/nqfw/duality-ai)
[![Inference](https://img.shields.io/badge/Inference-39.7ms-purple.svg)](https://github.com/nqfw/duality-ai)

**Desert Atlas AI** is an advanced offroad terrain understanding system developed for the **Duality AI Hackathon 2025**. It uses a deep ensemble of Vision Transformers (DINOv2) and Hierarchical Transformers (SegFormer) to segment complex desert and rocky environments for autonomous navigation.

---

## 🚀 Key Performance Metrics
| Metric | Result | Environment |
| :--- | :--- | :--- |
| **mIoU (Validation)** | **61.42%** | Synthetic Training Dist. |
| **mIoU (Official Test)** | **25.36%** | **Unseen/Novel Location** |
| **Inference Speed** | **39.7ms** | NVIDIA GPU (TTA included) |
| **Classes** | **10** | Trees, Bushes, Rocks, Sky, etc. |

---

## 🧠 Model Architecture: The "Turbo" Ensemble
Our system utilizes a high-performance ensemble strategy:
1. **DINOv2 ViT-B/14 (75% Weight):** Fine-tuned for domain-agnostic feature extraction and texture-level robustness.
2. **SegFormer B2 (25% Weight):** Provides structural consistency and global context.
3. **Test-Time Augmentation (TTA):** Horizontal flip and probability averaging yield a ~5% mIoU boost.
4. **Turbo Adaptation:** Backbone adaptation on the last 6 blocks of DINOv2 to bridge the synthetic-to-real gap.

---

## 📊 Research Data Dashboard
We built a professional **AI Research Dashboard** to interact with the model.
- **Visual Mapping:** High-precision segmentation masks with legend-overlay.
- **AI Insight:** Real-time terrain descriptions and obstacle detection feedback.
- **Telemetry Charts:** Live tracking of mIoU growth and inference latency.

---

## 🛠️ Getting Started

### 1. Requirements
- Python 3.11+
- CUDA-enabled GPU (for <50ms inference)

### 2. Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Run Inference Engine (Backend)
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 4. Open Research Dashboard
Simply open `web/index.html` in any modern browser to start interacting with the system.

---

## 📁 Repository Structure
- `api.py`: FastAPI backend for local model serving.
- `merge_models.py`: Core ensemble and TTA logic.
- `train_turbo_v3.py`: Ultra-fast domain adaptation script.
- `web/`: Full frontend codebase (HTML/JS/GSAP/Three.js).
- `*.pth`: Optimized model weights (Turbo & HailMary).

---

**Desert Atlas AI: Offroad Segmentation Baseline - 0.6142 mIoU**
This repository contains the optimized training pipeline for the DualityAI segmentation challenge.

📊 **Performance**
- **mIoU:** 0.6142 (Measured on validation set)
- **Official Test mIoU:** 0.2536 (Unseen Location)
- **Model:** DINOv2 ViT-B/14 + SegFormer Ensemble
- **Inference Speed:** 39.7ms (Optimized for Real-Time Deployment)

🚀 **Usage**
1. Install dependencies: `pip install -r requirements.txt`
2. Run training/inference: `python train_turbo_v3.py` or `python merge_models.py`
3. Web Dashboard: Open `web/index.html` via local server or direct file access.

🧠 **Model Weights**
The best optimized weights are saved as `segmentation_head_turbo.pth` and `dino_backbone_turbo.pth`.

---

Submission for **Duality AI 2025**.
