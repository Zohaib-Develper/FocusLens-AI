from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models
from pre_processing.preprocess_runtime import preprocess_base64_image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model architecture
model = models.resnet18(weights=None)  # No pretrained weights
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Load trained weights
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "attention_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        image_base64 = data.get("image")

        # print(image_base64)
        
        if not image_base64:
            return {"error": "No image provided"}
        
        # Preprocess
        img_tensor = preprocess_base64_image(image_base64)
        img_tensor = img_tensor.to(device)
        
        print(img_tensor.shape)
        # Model inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            
            print("probs", probs)
            print("pred_idx", pred_idx)
            print("confidence", confidence)
            # Labels: 0=Distracted, 1=Attentive
            prediction = "Attentive" if pred_idx == 1 else "Distracted"
        
        print(prediction)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "Distracted": float(probs[0]),
                "Attentive": float(probs[1])
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
