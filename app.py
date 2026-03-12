import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models.segmentation as segmentation
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MulticlassJaccardIndex
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

st.set_page_config(page_title="DualityAI Segmentation Training", layout="wide")

# ==========================================
# 1. Dummy Dataset with Albumentations
# ==========================================
class DummyDualityDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256), num_classes=2, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy image and mask
        image = np.random.randint(0, 256, (*self.img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, self.num_classes, self.img_size, dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask

# Albumentations transforms
def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==========================================
# 2. Models
# ==========================================
def get_resnet_model(num_classes):
    # Using DeepLabV3 with a ResNet50 backbone for segmentation
    model = segmentation.deeplabv3_resnet50(weights=segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_segformer_model(version="B2", num_classes=2):
    # Mapping version to config
    config_name = f"nvidia/segformer-b{version[-1]}-finetuned-cityscapes-1024-1024"
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            config_name, 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        st.warning(f"Could not load pretrained {version}. Using initialized weights. Error: {e}")
        config = SegformerConfig(num_labels=num_classes)
        model = SegformerForSemanticSegmentation(config)
    return model

# ==========================================
# 3. Training Loop MVP
# ==========================================
def train_model(model, train_loader, device, optimizer, criterion, metric_iou, epochs=1):
    model.train()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        metric_iou.reset()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            
            # Forward pass (Segformer returns a slightly different output format than torchvision)
            outputs = model(images)
            if hasattr(outputs, "logits"): # Segformer
                logits = outputs.logits
                # Resize logits to match mask size if needed
                logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            else: # ResNet/DeepLabV3
                logits = outputs['out']
                
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            metric_iou.update(preds, masks)
            
            # Update UI
            progress = (batch_idx + 1 + epoch * len(train_loader)) / (epochs * len(train_loader))
            progress_bar.progress(progress)
            
        iou_score = metric_iou.compute().item()
        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | IoU: {iou_score:.4f}")
        
    return model

# ==========================================
# 4. Visualizations
# ==========================================
def plot_predictions(image_tensor, true_mask, pred_mask):
    """Plot Original Image, Ground Truth, and Prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Denormalize image for viewing
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(true_mask.cpu().numpy(), cmap='jet')
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask.cpu().numpy(), cmap='jet')
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")
    
    return fig

def plot_confusion_matrix(cm_data, classes):
    """Plot Confusion Matrix using Seaborn."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig

# ==========================================
# Streamlit App UI
# ==========================================
def main():
    st.title("DualityAI Segmentation Trainer 🚀")
    
    with st.sidebar:
        st.header("Hyperparameters")
        model_choice = st.selectbox("Select Model", ["ResNet50 (DeepLabV3)", "SegFormer B2", "SegFormer B3"])
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=2)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=4)
        
        st.subheader("Learning Rates (AdamW)")
        lr_backbone = st.number_input("Backbone LR", value=1e-5, format="%e")
        lr_head = st.number_input("Head LR", value=1e-4, format="%e")
        
        start_training = st.button("Start Training", type="primary")

    if start_training:
        st.write("### Initializing Training...")
        
        # Device Selection (CUDA Enforcement)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            st.error("CUDA is not available! Training will fall back to CPU, but CUDA is required based on the specification.")
        else:
            st.success(f"Training on: {torch.cuda.get_device_name(0)} (CUDA)")
            
        num_classes = 2 # Background + Object
        
        # Load Dataset MVP
        transform = get_transforms((256, 256))
        train_dataset = DummyDualityDataset(num_samples=20, num_classes=num_classes, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Load Model
        with st.spinner(f"Loading {model_choice}..."):
            if "ResNet" in model_choice:
                model = get_resnet_model(num_classes)
                # Split params for differential learning rates
                backbone_params = list(model.backbone.parameters())
                head_params = list(model.classifier.parameters())
            else:
                version = "B2" if "B2" in model_choice else "B3"
                model = get_segformer_model(version, num_classes)
                # Transformers segregation (Segformer doesn't have a direct 'backbone' property named explicitly sometimes, assuming standard names)
                if hasattr(model, 'segformer'): 
                    backbone_params = list(model.segformer.parameters())
                    head_params = list(model.decode_head.parameters())
                else: # Fallback single LR config
                    backbone_params = list(model.parameters())
                    head_params = []
                    
            model.to(device)
            
        # AdamW Optimizer with different learning rates
        optimizer_grouped_parameters = [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head} if len(head_params) > 0 else {}
        ]
        # Remove empty dict if head_params was empty
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if 'params' in g]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        criterion = nn.CrossEntropyLoss()
        
        # Torchmetrics IoU
        metric_iou = MulticlassJaccardIndex(num_classes=num_classes).to(device)

        st.write("### Training Progress")
        model = train_model(model, train_loader, device, optimizer, criterion, metric_iou, epochs=epochs)
        
        st.success("Training Complete!")
        
        # MVP Visualizations Placeholder
        st.write("### Evaluation Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Sample Prediction vs Ground Truth")
            # Get a single batch for vis
            images, masks = next(iter(train_loader))
            images, masks = images.to(device), masks.to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(images[0:1])
                if hasattr(outputs, "logits"):
                    logits = nn.functional.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    logits = outputs['out']
                pred_mask = torch.argmax(logits, dim=1)[0]
                
            fig_pred = plot_predictions(images[0], masks[0], pred_mask)
            st.pyplot(fig_pred)
            
        with col2:
            st.write("#### Confusion Matrix (Dummy Data)")
            # Generating dummy CM data for visualization testing
            dummy_cm = np.random.randint(10, 100, size=(num_classes, num_classes))
            fig_cm = plot_confusion_matrix(dummy_cm, [f"Class {i}" for i in range(num_classes)])
            st.pyplot(fig_cm)
            
        st.write("#### Failing Cases (Low IoU)")
        st.info("Failing cases view will be available once the real dataset is mounted and evaluated.")

if __name__ == "__main__":
    main()
