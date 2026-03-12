import os
import numpy as np
from PIL import Image
from tqdm import tqdm

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def analyze():
    mask_dir = r"C:\Users\lenovo\OneDrive\Desktop\DualityAI\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    files = os.listdir(mask_dir)[:300] # Check 300 samples
    counts = np.zeros(10)
    
    for f in tqdm(files):
        mask = np.array(Image.open(os.path.join(mask_dir, f)))
        for val, idx in value_map.items():
            counts[idx] += np.sum(mask == val)
            
    weights = 1.0 / (counts + 1e-6)
    weights = weights / np.sum(weights) * 10
    print("Counts:", counts)
    print("Recommended Weights:", weights.tolist())

if __name__ == "__main__":
    analyze()
