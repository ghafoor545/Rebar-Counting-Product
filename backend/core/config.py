import os

# Base directory of the project (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# core/config.py -> core -> backend -> Rebar-Counting

# ONNX model path: now points to backend/models/yolo11s.onnx
MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "epoch20.onnx")

APP_SECRET = "change-this-secret"

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")
DET_DIR = os.path.join(DATA_DIR, "detections")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
SESSION_FILE = os.path.join(DATA_DIR, "session.json")

PER_PAGE = 8

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DET_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
