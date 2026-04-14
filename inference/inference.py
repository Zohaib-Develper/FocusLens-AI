"""
inference.py

Load trained model and predict attention on a single image.
Supports:
- Any .jpg/.png
- Real-time confidence scores
- Visual output with label
"""

import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

# =============================
# CONFIG
# =============================
# Get the script's directory and construct absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from inference/

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "attention_model.pth")
IMG_PATH   = os.path.join(PROJECT_ROOT, "image3.jpg")  # ← CHANGE THIS TO YOUR IMAGE

# Labels
LABELS = ["Distracted", "Attentive"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================
# LOAD MODEL
# =============================
print("Loading model...")
model = models.resnet18(weights=None)  # No pretrained weights
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# =============================
# PREPROCESS IMAGE
# =============================
def preprocess_image(img_path):
    # Read with OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Resize to 224x224
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img.astype(np.float32) / 255.0

    # To tensor: (1, 1, 224, 224)
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    return img

# =============================
# PREDICT
# =============================
print(f"Predicting on: {IMG_PATH}")
input_tensor = preprocess_image(IMG_PATH)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

# =============================
# DISPLAY RESULT
# =============================
print("\n" + "="*50)
print(" PREDICTION RESULT ")
print("="*50)
print(f"Prediction:   {LABELS[pred_idx]}")
print(f"Confidence:   {confidence:.1%}")
print(f"Probabilities:")
print(f"  Distracted: {probs[0]:.1%}")
print(f"  Attentive:  {probs[1]:.1%}")
print("="*50)

# Optional: Show image with label
img_bgr = cv2.imread(IMG_PATH)
img_bgr = cv2.resize(img_bgr, (300, 300))
color = (0, 255, 0) if pred_idx == 1 else (0, 0, 255)
cv2.putText(img_bgr, f"{LABELS[pred_idx]} ({confidence:.0%})", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
cv2.imshow("Prediction", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()