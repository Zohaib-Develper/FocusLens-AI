"""
train.py

Training script for preprocessed dataset.
Uses images from ./data/processed_dataset/

"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import from utils.py file
import importlib.util
spec = importlib.util.spec_from_file_location("utils", os.path.join(PROJECT_ROOT, "utils", "utils.py"))
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)
FaceAttentionDataset = utils_module.FaceAttentionDataset
get_transform = utils_module.get_transform

# =============================
# CONFIG
# =============================
# Get the script's directory and construct absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from training/

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_dataset")  # Preprocessed data
TRAIN_CSV = os.path.join(PROCESSED_DIR, "train_data.csv")
TEST_CSV  = os.path.join(PROCESSED_DIR, "test_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "attention_model.pth")

BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LEARNING_RATE = 0.001
PATIENCE      = 5

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================
# DATA LOADERS
# =============================
train_dataset = FaceAttentionDataset(TRAIN_CSV, PROCESSED_DIR, transform=get_transform())
test_dataset  = FaceAttentionDataset(TEST_CSV,  PROCESSED_DIR, transform=get_transform())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

# Display dataset statistics
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Total subjects: {train_df['subject'].nunique()}")
print(f"Total images: {len(train_dataset) + len(test_dataset)}")
print(f"\nTrain set: {len(train_dataset)} images")
print(f"  - Attentive (1):   {(train_df['label']==1).sum()}")
print(f"  - Distracted (0):  {(train_df['label']==0).sum()}")
print(f"\nTest set: {len(test_dataset)} images")
print(f"  - Attentive (1):   {(test_df['label']==1).sum()}")
print(f"  - Distracted (0):  {(test_df['label']==0).sum()}")
print("="*60)

# =============================
# MODEL: ResNet18 (grayscale → 1 channel)
# =============================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 1-channel input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Binary output
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

# =============================
# LOSS, OPTIMIZER, SCHEDULER
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4
)

# =============================
# TRAINING LOOP + EARLY STOPPING
# =============================
best_val_acc = 0.0
patience_counter = 0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n" + "="*60)
print("STARTING TRAINING (MediaPipe Dataset)")
print("="*60)

for epoch in range(NUM_EPOCHS):
    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0
    correct = total = 0

    # Progress bar for training batches
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", 
                      leave=False, ncols=100)
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = outputs.max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        # Update progress bar with current metrics
        current_acc = 100.0 * correct / total
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

    train_loss /= len(train_loader)
    train_acc   = 100.0 * correct / total

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    correct = total = 0
    
    # Progress bar for validation batches
    val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ", 
                    leave=False, ncols=100)
    
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = outputs.max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            # Update progress bar with current metrics
            current_acc = 100.0 * correct / total
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

    val_loss /= len(test_loader)
    val_acc   = 100.0 * correct / total

    # ---- record ----
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # ---- scheduler step ----
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr != LEARNING_RATE * (0.5 ** (epoch // 3)):
        print(f"   Learning rate reduced to {current_lr:.6f}")

    # ---- print epoch summary ----
    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}%")

    # ---- save best model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"   → NEW BEST MODEL (val acc {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1

    # ---- early stopping ----
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch+1}")
        break

print("="*60)
print("TRAINING FINISHED")
print("="*60)

# Load the best checkpoint
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =============================
# FINAL TEST EVALUATION
# =============================
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, pred = outputs.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nCLASSIFICATION REPORT")
print(classification_report(all_labels, all_preds,
                            target_names=["Distracted", "Attentive"]))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Distracted', 'Attentive'],
            yticklabels=['Distracted', 'Attentive'])
plt.title('Confusion Matrix (MediaPipe)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_mediapipe.png"))
plt.show()

# Training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"],   label="Val")
plt.title("Loss (MediaPipe)")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history["train_acc"], label="Train")
plt.plot(history["val_acc"],   label="Val")
plt.title("Accuracy % (MediaPipe)")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves_mediapipe.png"))
plt.show()

print(f"\nBest model saved at: {MODEL_PATH}")
print(f"Plots saved in: {MODEL_DIR}")
