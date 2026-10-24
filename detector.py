from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Put best.pt inside the project's models folder."
    )

model = YOLO(str(MODEL_PATH))


def detect_image(image, confidence: float = 0.25):
    return model.predict(source=image, conf=confidence, verbose=False)[0]


def get_class_names():
    return model.names
