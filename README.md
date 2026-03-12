# DualityAI Offroad Segmentation Baseline - 0.4991 mIoU

This repository contains the optimized training pipeline for the DualityAI segmentation challenge.

## 📊 Performance
- **mIoU:** 0.4991 (Measured on `val/` set)
- **Model:** SegFormer-B0
- **Training Time:** ~12 minutes
- **Hardware:** Optimized for 6GB VRAM (AMP + 320x320)

## 🚀 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run training/inference: `python train_segmentation.py`
3. Predictions are saved to `test_predictions/`

## 🧠 Model Weights
The best weights are saved in `best_segformer.pth`.
