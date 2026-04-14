"""
preprocess_runtime.py

Preprocessing utilities for the FastAPI backend.
Handles base64 image decoding and preprocessing for model inference.
Uses MediaPipe face detection to match training preprocessing pipeline.
"""

import torch
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import mediapipe as mp

# MediaPipe configuration (matching training settings)
TARGET_SIZE = (224, 224)
PADDING_RATIO = 0.4
MIN_DETECTION_CONFIDENCE = 0.5
MODEL_SELECTION = 0

# Initialize MediaPipe Face Detection (singleton pattern for efficiency)
mp_face_detection = mp.solutions.face_detection
_face_detection = None

def get_face_detection():
    """Get or create MediaPipe face detection instance."""
    global _face_detection
    if _face_detection is None:
        _face_detection = mp_face_detection.FaceDetection(
            model_selection=MODEL_SELECTION,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )
    return _face_detection


def preprocess_base64_image(image_base64: str) -> torch.Tensor:
    """
    Convert base64 image to preprocessed tensor for model inference.
    Uses MediaPipe face detection to match training preprocessing.
    
    Args:
        image_base64: Base64 encoded image string (with or without data URI prefix)
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 1, 224, 224)
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Convert to PIL Image then to OpenCV format
        image = Image.open(BytesIO(image_bytes))
        img = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        h, w = img.shape[:2]
        
        # ------------------------------------------------------------------
        # MediaPipe Face Detection (matching training pipeline)
        # ------------------------------------------------------------------
        face_detection = get_face_detection()
        
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        results = face_detection.process(img_rgb)
        
        if results.detections:
            # Get the first (most confident) detection
            detection = results.detections[0]
            
            # Get bounding box (normalized coordinates)
            bboxC = detection.location_data.relative_bounding_box
            
            # Convert to pixel coordinates
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)
            
            # Calculate padding (40% as in training)
            pad_w = int(box_w * PADDING_RATIO)
            pad_h = int(box_h * PADDING_RATIO)
            
            # Expand bounding box with padding
            x1 = max(x - pad_w, 0)
            y1 = max(y - pad_h, 0)
            x2 = min(x + box_w + pad_w, w)
            y2 = min(y + box_h + pad_h, h)
            
            # Make it square by expanding the smaller dimension
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if crop_w > crop_h:
                # Expand height
                diff = crop_w - crop_h
                y1 = max(y1 - diff // 2, 0)
                y2 = min(y2 + diff // 2, h)
            elif crop_h > crop_w:
                # Expand width
                diff = crop_h - crop_w
                x1 = max(x1 - diff // 2, 0)
                x2 = min(x2 + diff // 2, w)
            
            # Extract face region
            face_roi = img[y1:y2, x1:x2]
            
        else:
            # ------------------------------------------------------------------
            # Fallback: Smart center crop (matching training fallback)
            # ------------------------------------------------------------------
            crop_size = int(min(h, w) * 0.8)
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            
            x1 = max(cx - half, 0)
            y1 = max(cy - half, 0)
            x2 = min(cx + half, w)
            y2 = min(cy + half, h)
            
            face_roi = img[y1:y2, x1:x2]
        
        # ------------------------------------------------------------------
        # Convert to grayscale and resize (matching training pipeline)
        # ------------------------------------------------------------------
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Ensure minimum size
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            # Last resort: use entire image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Resize with high-quality interpolation (matching training)
        if gray.shape[0] > TARGET_SIZE[0] or gray.shape[1] > TARGET_SIZE[1]:
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        img_normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor: (1, 1, 224, 224)
        img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")
