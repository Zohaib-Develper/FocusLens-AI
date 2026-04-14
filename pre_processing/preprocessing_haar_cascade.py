"""
preprocess.py

Preprocesses raw face images from nested subject folders:
1. Recursively scans all subdirectories for images
2. Parses filenames to extract pose/illumination metadata
3. Detects and crops faces (with padding to preserve head tilt)
4. Resizes to 224x224, converts to grayscale, normalizes
5. Saves processed images (flattened structure)
6. Generates attention labels based on yaw/pitch/head thresholds
7. Splits into train/test (stratified by label)

Output:
- ./data/processed_dataset/
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

# =============================
# CONFIGURATION
# =============================

# Directories
DATA_DIR = "./data/dataset"                    # Raw images (with subdirectories)
OUTPUT_DIR = "./data/processed_dataset"            # Preprocessed output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing settings
TARGET_SIZE = (224, 224)
PADDING_RATIO = 0.4                                # Increased padding for more space

# Labeling thresholds (tune these!)
YAW_THRESH = 10
PITCH_THRESH = 10
HEAD_THRESH = 10

# Filename pattern: 0001_2m_0P_15V_5H.jpg → subject, illum, pitch, yaw, head
FILENAME_PATTERN = re.compile(r"(\d{4})_(\d+m)_(-?\d+)P_(-?\d+)V_(-?\d+)H")


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
                "src_path": src_path,  # Full path to source file
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
# STEP 2: Preprocess & Save Images
# =============================
def preprocess_and_save_images(df):
    """
    Improved preprocessing with:
    - Multiple face detection strategies
    - Better padding to preserve space around faces
    - Square aspect ratio preservation
    - Robust fallback mechanisms
    """
    # Load multiple cascades for better detection
    cascade_front = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml')
    cascade_alt = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    if cascade_front.empty():
        raise RuntimeError("OpenCV frontal cascade file missing!")

    processed = 0
    detected = 0
    fallback_count = 0

    print("Preprocessing images with improved face detection...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        src_path = row["src_path"]  # Use full source path
        dst_path = os.path.join(OUTPUT_DIR, row["filename"])

        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Cannot read {row['filename']}, skipping...")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ------------------------------------------------------------------
        # Multi-strategy face detection
        # ------------------------------------------------------------------
        faces = []
        
        # Strategy 1: Standard frontal detection
        faces = cascade_front.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        # Strategy 2: Profile detection (for turned heads)
        if len(faces) == 0 and not cascade_profile.empty():
            faces = cascade_profile.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        
        # Strategy 3: Alternative frontal cascade
        if len(faces) == 0 and not cascade_alt.empty():
            faces = cascade_alt.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        
        # Strategy 4: More permissive settings
        if len(faces) == 0:
            faces = cascade_front.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))

        # ------------------------------------------------------------------
        # Extract face region with proper padding
        # ------------------------------------------------------------------
        if len(faces) > 0:
            # Use the largest detected face
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            
            # Calculate generous padding (40% of face size)
            pad_w = int(fw * PADDING_RATIO)
            pad_h = int(fh * PADDING_RATIO)
            
            # Expand bounding box with padding
            x1 = max(fx - pad_w, 0)
            y1 = max(fy - pad_h, 0)
            x2 = min(fx + fw + pad_w, w)
            y2 = min(fy + fh + pad_h, h)
            
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
            
            face_roi = gray[y1:y2, x1:x2]
            detected += 1
        else:
            # ------------------------------------------------------------------
            # Fallback: Smart center crop with generous margins
            # ------------------------------------------------------------------
            # Use 80% of the smaller dimension to ensure we capture the face
            crop_size = int(min(h, w) * 0.8)
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            
            x1 = max(cx - half, 0)
            y1 = max(cy - half, 0)
            x2 = min(cx + half, w)
            y2 = min(cy + half, h)
            
            face_roi = gray[y1:y2, x1:x2]
            fallback_count += 1

        # ------------------------------------------------------------------
        # Ensure minimum size and resize
        # ------------------------------------------------------------------
        if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            # Last resort: use entire image
            face_roi = gray
            fallback_count += 1

        # Resize to target size with high-quality interpolation
        if face_roi.shape[0] > TARGET_SIZE[0] or face_roi.shape[1] > TARGET_SIZE[1]:
            # Downsampling: use INTER_AREA for best quality
            resized = cv2.resize(face_roi, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        else:
            # Upsampling: use INTER_CUBIC for smoothness
            resized = cv2.resize(face_roi, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

        # Normalize and save
        normalized = resized.astype(np.float32) / 255.0
        cv2.imwrite(dst_path, (normalized * 255).astype(np.uint8))
        
        processed += 1

    print(f"\nPreprocessing complete:")
    print(f"  ✓ Total processed: {processed}/{len(df)}")
    print(f"  ✓ Face detected: {detected} ({100*detected/processed:.1f}%)")
    print(f"  ⚠ Fallback used: {fallback_count} ({100*fallback_count/processed:.1f}%)")


# =============================
# STEP 3: Generate Labels
# =============================
def generate_labels(df):
    print("Generating attention labels...")
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
    
    # Drop src_path column before saving (not needed for training)
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
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)

    # 1. Parse (recursively scan all subdirectories)
    df = parse_filenames()
    if df.empty:
        raise ValueError("No valid images found. Check DATA_DIR and filename pattern.")

    # 2. Preprocess images
    preprocess_and_save_images(df)

    # 3. Label
    df = generate_labels(df)

    # 4. Split & Save
    split_and_save(df)

    print("\nPREPROCESSING COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()