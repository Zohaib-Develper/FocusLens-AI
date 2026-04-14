# FocusLens AI

Real-time attention monitoring for Google Meet powered by deep learning. Uses a custom ResNet18 model to classify participant attentiveness from video frames, with results displayed as live overlays in the meeting.

## Architecture

```
Chrome Extension (content.js)
    ↓ captures video frames every 3s
    ↓ base64 JPEG via POST
FastAPI Backend (app.py)
    ↓ MediaPipe face detection + preprocessing
    ↓ PyTorch ResNet18 inference
    ↓ returns prediction + confidence
Chrome Extension
    ↓ renders overlay on each participant tile
    └── Attentive (green glow) / Distracted (red glow)
```

## How It Works

1. A Chrome extension injects into Google Meet and captures video frames from participant tiles
2. Frames are sent to a local FastAPI server running the trained PyTorch model
3. The model predicts whether each participant is **Attentive** or **Distracted**
4. Results are displayed as glassmorphism overlays with confidence scores directly on the Meet UI

## Tech Stack

- **Model**: ResNet18 (modified for grayscale input) — PyTorch
- **Face Detection**: MediaPipe Face Detection
- **Backend**: FastAPI + Uvicorn
- **Extension**: Chrome Manifest V3 content script
- **Dataset**: [Columbia Gaze Dataset](https://ceal.cs.columbia.edu/columbiagaze/) (~5,600 images, 56 subjects)

## Project Structure

```
FocusLens-AI/
├── extension/                    # Chrome extension
│   ├── manifest.json            # Manifest V3 config
│   ├── content.js               # Frame capture + API + overlay logic
│   ├── styles.css               # Overlay styling
│   └── icons/                   # Extension icons
├── pre_processing/
│   ├── preprocessing_mediapipe.py    # Dataset preprocessing with MediaPipe
│   ├── preprocessing_haar_cascade.py # Alternative Haar cascade approach
│   └── preprocess_runtime.py         # Runtime preprocessing for inference
├── training/
│   └── train.py                 # Model training script
├── inference/
│   └── inference.py             # Standalone inference testing
├── models/
│   ├── attention_model.pth      # Trained model weights
│   ├── confusion_matrix_mediapipe.png
│   └── training_curves_mediapipe.png
├── utils/
│   └── utils.py                 # Dataset loader
├── app.py                       # FastAPI server
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+ (MediaPipe compatibility)
- Google Chrome
- pip

### 1. Clone and install dependencies

```bash
git clone https://github.com/Zohaib-Develper/FocusLens-AI.git
cd FocusLens-AI
pip install -r requirements.txt
```

### 2. Start the backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at `http://localhost:8000`

### 3. Load the Chrome extension

1. Open `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder from this project

### 4. Test it

1. Join a Google Meet call
2. The extension automatically starts analyzing video frames
3. You'll see overlay labels on each participant showing:
   - **Status**: "Attentive" or "Distracted" with emoji indicator
   - **Confidence bar**: real-time prediction confidence
   - **Probability breakdown**: attentive vs distracted percentages
   - **Visual glow**: green for attentive, red for distracted

## Model Details

### Preprocessing

All images preprocessed using MediaPipe Face Detection:
- Resized to **224×224** grayscale
- Face ROI extraction with 40% padding
- Min-max normalization to [0, 1]
- Square crop with aspect ratio preservation

### Training

| Parameter | Value |
|-----------|-------|
| Base Model | ResNet18 (ImageNet pretrained, fine-tuned) |
| Input | 224 × 224 × 1 (grayscale) |
| Output | Binary (Attentive / Distracted) |
| Optimizer | Adam (lr=0.001) |
| Batch Size | 32 |
| Early Stopping | 5 epochs patience |
| Training stopped | Epoch 8 (converged) |

### Results

- **Test Accuracy**: 82.34%
- Training curves and confusion matrix available in `/models`

## API

### POST /predict

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
  "prediction": "Attentive",
  "confidence": 0.95,
  "probabilities": {
    "Distracted": 0.05,
    "Attentive": 0.95
  }
}
```

## Author

**Zohaib Musharaf**  
GitHub: [Zohaib-Develper](https://github.com/Zohaib-Develper)
