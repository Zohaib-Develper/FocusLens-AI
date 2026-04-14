"""
pre-processing-mediapipe.py

Advanced preprocessing using MediaPipe Face Detection for superior accuracy.
MediaPipe can detect faces in challenging conditions:
- Extreme head poses (±90° yaw/pitch)
- Poor lighting conditions
- Partial occlusions
- Profile views

Advantages over Haar Cascades:
✓ 95-99% detection rate (vs 85-92% for Haar)
✓ Better handling of rotated faces
✓ More accurate bounding boxes
✓ Faster processing with GPU support

Output:
- ./data/processed_dataset_mediapipe/
    ├── *.jpg (preprocessed)
    ├── labeled_data.csv
    ├── train_data.csv
    └── test_data.csv
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mediapipe as mp

# =============================
# CONFIGURATION
# =============================

# Get the script's directory and construct absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from pre-processing/

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")                    # Raw images (with subdirectories)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_dataset")        # Preprocessed output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing settings
TARGET_SIZE = (224, 224)
PADDING_RATIO = 0.4                                   # 40% padding around face

# Labeling thresholds (tune these!)
YAW_THRESH   = 12    # left-right movement
PITCH_THRESH = 23    # up-down movement
HEAD_THRESH  = 12    # tilt

# Filename pattern: 0001_2m_0P_15V_5H.jpg → subject, illum, pitch, yaw, head
FILENAME_PATTERN = re.compile(r"(\d{4})_(\d+m)_(-?\d+)P_(-?\d+)V_(-?\d+)H")

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5    # Lower = more detections, more false positives
MODEL_SELECTION = 0               # 0 = within 2m, 1 = within 5m (use 0 for close-up faces)


# =============================
# STEP 1: Parse Filenames (Recursive)
# =============================
def parse_filenames():
    data = []
    print("Parsing filenames from all subdirectories...")
    
    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            match = FILENAME_PATTERN.match(filename)
            if not match:
                # Skip non-matching files (like .DS_Store)
                continue
            
            subject, illum, pitch, yaw, head = match.groups()
            
            # Store full source path for reading
            src_path = os.path.join(root, filename)
            
            data.append({
                "filename": filename,
                "src_path": src_path,
                "subject": int(subject),
                "illumination": illum,
                "pitch": int(pitch),
                "yaw": int(yaw),
                "head": int(head)
            })
    
    df = pd.DataFrame(data)
    print(f"Found {len(df)} valid images across {df['subject'].nunique()} subjects.")
    return df


# =============================
# STEP 2: Preprocess with MediaPipe
# =============================
def preprocess_and_save_images(df):
    """
    Advanced preprocessing using MediaPipe Face Detection:
    - Superior face detection accuracy (95-99%)
    - Handles extreme poses and lighting
    - Provides confidence scores
    - GPU-accelerated (if available)
    """
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    
    print(f"Initializing MediaPipe Face Detection...")
    print(f"  - Model: {'Short-range (0-2m)' if MODEL_SELECTION == 0 else 'Full-range (0-5m)'}")
    print(f"  - Min confidence: {MIN_DETECTION_CONFIDENCE}")
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=MODEL_SELECTION,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )
    
    processed = 0
    detected = 0
    fallback_count = 0
    confidence_scores = []
    
    print("\nPreprocessing images with MediaPipe Face Detection...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        src_path = row["src_path"]
        dst_path = os.path.join(OUTPUT_DIR, row["filename"])
        
        # Read image
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Cannot read {row['filename']}, skipping...")
            continue
        
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # ------------------------------------------------------------------
        # MediaPipe Face Detection
        # ------------------------------------------------------------------
        results = face_detection.process(img_rgb)
        
        if results.detections:
            # Get the first (most confident) detection
            detection = results.detections[0]
            confidence = detection.score[0]
            confidence_scores.append(confidence)
            
            # Get bounding box (normalized coordinates)
            bboxC = detection.location_data.relative_bounding_box
            
            # Convert to pixel coordinates
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)
            
            # Calculate padding
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
            detected += 1
            
        else:
            # ------------------------------------------------------------------
            # Fallback: Smart center crop
            # ------------------------------------------------------------------
            crop_size = int(min(h, w) * 0.8)
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            
            x1 = max(cx - half, 0)
            y1 = max(cy - half, 0)
            x2 = min(cx + half, w)
            y2 = min(cy + half, h)
            
            face_roi = img[y1:y2, x1:x2]
            fallback_count += 1
        
        # ------------------------------------------------------------------
        # Convert to grayscale and resize
        # ------------------------------------------------------------------
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Ensure minimum size
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            # Last resort: use entire image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            fallback_count += 1
        
        # Resize with high-quality interpolation
        if gray.shape[0] > TARGET_SIZE[0] or gray.shape[1] > TARGET_SIZE[1]:
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Normalize and save
        normalized = resized.astype(np.float32) / 255.0
        cv2.imwrite(dst_path, (normalized * 255).astype(np.uint8))
        
        processed += 1
    
    # Cleanup
    face_detection.close()
    
    # Statistics
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Total processed: {processed}/{len(df)}")
    print(f"✓ Face detected: {detected} ({100*detected/processed:.1f}%)")
    print(f"⚠ Fallback used: {fallback_count} ({100*fallback_count/processed:.1f}%)")
    
    if confidence_scores:
        print(f"\nDetection Confidence Statistics:")
        print(f"  - Mean: {np.mean(confidence_scores):.3f}")
        print(f"  - Min:  {np.min(confidence_scores):.3f}")
        print(f"  - Max:  {np.max(confidence_scores):.3f}")
    
    print(f"{'='*60}")


# =============================
# STEP 3: Generate Labels
# =============================
def generate_labels(df):
    print("\nGenerating attention labels...")
    df["label"] = df.apply(
        lambda r: 1 if (
            abs(r["yaw"]) <= YAW_THRESH and
            abs(r["pitch"]) <= PITCH_THRESH and
            abs(r["head"]) <= HEAD_THRESH
        ) else 0,
        axis=1
    )
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
    print(f"\nClass balance:")
    print(f"  Attentive (1):   {(df['label']==1).sum()} ({100*(df['label']==1).sum()/len(df):.1f}%)")
    print(f"  Distracted (0):  {(df['label']==0).sum()} ({100*(df['label']==0).sum()/len(df):.1f}%)")
    return df


# =============================
# STEP 4: Train/Test Split
# =============================
def split_and_save(df):
    print("\nSplitting into train/test...")
    
    # Drop src_path column before saving
    df_to_save = df.drop(columns=['src_path'])
    
    train_df, test_df = train_test_split(
        df_to_save,
        test_size=0.2,
        random_state=42,
        stratify=df_to_save["label"]
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    
    # Save
    df_to_save.to_csv(os.path.join(OUTPUT_DIR, "labeled_data.csv"), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)
    
    print(f"All CSVs saved in {OUTPUT_DIR}")
    return train_df, test_df


# =============================
# MAIN EXECUTION
# =============================
def main():
    print("="*60)
    print("MEDIAPIPE PREPROCESSING PIPELINE")
    print("="*60)
    print("Using MediaPipe for superior face detection accuracy")
    print("="*60)
    
    # 1. Parse (recursively scan all subdirectories)
    df = parse_filenames()
    if df.empty:
        raise ValueError("No valid images found. Check DATA_DIR and filename pattern.")
    
    # 2. Preprocess images with MediaPipe
    preprocess_and_save_images(df)
    
    # 3. Label
    df = generate_labels(df)
    
    # 4. Split & Save
    split_and_save(df)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
