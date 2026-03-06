from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse

from dotenv import load_dotenv
import cv2
import time

from backend.services.detector import RebarBundleDetector, img_to_data_uri, model
from backend.services.oak_utils import grab_oak_frame, close_oak_device
from backend.api import auth_routes, detection_routes
from backend.db import init_db

load_dotenv()

app = FastAPI(title="Rebar-Counting API")

# CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OAK session
OAK_SESSION = {}

# Detection service
detector_service = RebarBundleDetector(
    eps=100.0,
    min_bundle_size=5,
    min_samples=2,
    row_tolerance=40.0,
    use_adaptive_eps=True,
)

# Simple detection router
simple_router = APIRouter(tags=["simple-detection"])


@simple_router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Basic image upload detection endpoint.
    """
    contents = await file.read()
    result = detector_service.detect_image(contents)
    return result


@simple_router.post("/oak-d")
def detect_oak_d():
    """
    One-shot OAK-D detection / debug endpoint.
    """
    frame, grab_error = grab_oak_frame(OAK_SESSION, wait_sec=2.0)
    if grab_error or frame is None:
        return {
            "count": 0,
            "error": grab_error or "Failed to grab frame from OAK-D Pro.",
            "bundle_info": None,
            "image": None,
        }

    result = detector_service.detect_oak_camera(frame)
    return result


@simple_router.get("/oak-snapshot")
def oak_snapshot():
    """
    Returns a single raw JPEG frame from the OAK-D camera.
    """
    frame, err = grab_oak_frame(OAK_SESSION, wait_sec=2.0)
    if err or frame is None:
        return Response(status_code=503)

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return Response(status_code=500)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


def oak_mjpeg_generator():
    """
    Generator for MJPEG streaming from OAK-D.
    """
    try:
        while True:
            frame, err = grab_oak_frame(OAK_SESSION, wait_sec=2.0)
            if frame is None:
                time.sleep(0.05)
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
    finally:
        # Device closed on shutdown in on_shutdown()
        pass


@simple_router.get("/oak-stream")
def oak_stream():
    """
    OAK-D live preview stream (NO DETECTION).
    """
    return StreamingResponse(
        oak_mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Attach routers
app.include_router(simple_router)
app.include_router(auth_routes.router)
app.include_router(detection_routes.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.on_event("shutdown")
def on_shutdown():
    close_oak_device(OAK_SESSION)


@app.get("/")
def root():
    return {"message": "Rebar-Counting backend is running"}