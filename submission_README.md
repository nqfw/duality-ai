# Duality AI Hackathon: Offroad Semantic Segmentation
## Team Submission Package

### Overview
This submission features a robust ensemble of **DINOv2 (ViT-B/14)** and **Segformer B2**. By leveraging weighted probability blending and domain-specific fine-tuning, we achieve high accuracy and stable performance across offroad desert environments.

### Performance Summary
- **mIoU (Validation)**: 0.6142
- **Inference Latency**: 39.69ms (on NVIDIA GPU via `cu118`)
- **Classes Supported**: 10 (100: Trees, 200: Lush Bushes, 300: Dry Grass, 500: Dry Bushes, 550: Ground Clutter, 600: Flowers, 700: Logs, 800: Rocks, 7100: Landscape, 10000: Sky)

### Dependencies
- Python 3.11
- PyTorch 2.0+ (with CUDA)
- Transformers
- Pillow, NumPy, Matplotlib

### How to Reproduce
1. Ensure `segmentation_head_robust.pth` and `best_b2.pth` are in the root directory.
2. Run the metrics generation script:
   ```bash
   python generate_report_metrics.py
   ```

### Methodology
We utilized **Test-Time Augmentation (TTA)** and probability blending to reduce noise in individual model predictions. For domain shift resilience, we applied heavy color jitter and blur during a specialized robustness fine-tuning pass.
