"""
DeepGuard - inference.
This is the file the rest of the team imports:

    from model.inference import predict
    result = predict("photo.jpg")
    # {"verdict": "FAKE", "confidence": 87.3}   or   {"error": "..."}

predict() accepts a file path, an uploaded file object (e.g. Streamlit), or a PIL image.
"""

import os

import timm
import torch
from PIL import Image
from torchvision import transforms

# Weights live in model/weights/detector.pth (not stored in git, too large).
# Override the location with the DEEPGUARD_WEIGHTS environment variable if needed.
WEIGHTS_PATH = os.environ.get(
    "DEEPGUARD_WEIGHTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "detector.pth"),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None  # loaded once on first use, then reused for every request


def load_model():
    global _model
    if _model is None:
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        _model = model
    return _model


def predict(image):
    try:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB")
    except Exception:
        return {"error": "Could not read the file as an image."}

    try:
        model = load_model()
    except FileNotFoundError:
        return {"error": f"Model weights not found at {WEIGHTS_PATH}"}

    x = _TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    label = int(probs.argmax())  # 0 = real, 1 = fake
    return {
        "verdict": "FAKE" if label == 1 else "REAL",
        "confidence": round(probs[label].item() * 100, 2),
    }
