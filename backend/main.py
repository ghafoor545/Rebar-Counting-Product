# backend/main.py

from fastapi import FastAPI, UploadFile, File, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from dotenv import load_dotenv
import cv2
import time
import numpy as np
import threading
from collections import deque

from backend.services.detector import RebarBundleDetector, img_to_data_uri, model
from backend.services.oak_utils import grab_oak_frame, close_oak_device
from backend.api import auth_routes, detection_routes
from backend.db import init_db
from backend.api.detection_routes import record_detection

load_dotenv()

app = FastAPI(title="Rebar-Counting API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OAK session
OAK_SESSION = {}

# Latest detection result cache (for live feed display - ONLY distances, no count)
latest_detection_result = {
    "bundles": [],
    "nearest_bundle": None,
    "timestamp": 0
}

# Detector for live feed - ONLY shows bundles and distances, NO counting
live_feed_detector = RebarBundleDetector(
    eps=100.0,
    min_bundle_size=3,
    min_samples=2,
    row_tolerance=40.0,
    use_adaptive_eps=True,
    use_depth_filter=True,
    max_detection_distance_mm=1500.0,
    min_detection_distance_mm=200.0,
    track_bundle_distances=True,
    nearest_bundle_only=False,
    draw_seg_masks=False,
    debug=False,
)

# Detector for capture - counts ALL bundles but returns individual counts
capture_detector = RebarBundleDetector(
    eps=100.0,
    min_bundle_size=3,
    min_samples=2,
    row_tolerance=40.0,
    use_adaptive_eps=True,
    use_depth_filter=True,
    max_detection_distance_mm=1500.0,
    min_detection_distance_mm=200.0,
    track_bundle_distances=True,
    nearest_bundle_only=False,  # Get all bundles, we'll decide which to count
    draw_seg_masks=True,
    debug=False,
)

# Simple detection router
simple_router = APIRouter(tags=["simple-detection"])

# Background thread for continuous detection on live feed
detection_thread_running = True
detection_interval = 0.5


def continuous_detection():
    """Background thread that continuously processes frames to get bundle distances only"""
    global latest_detection_result, detection_thread_running
    
    print("[Continuous Detection] Started - Calculating distances for all bundles")
    
    while detection_thread_running:
        try:
            # Get latest frame
            frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=0.1)
            
            if err or frame is None:
                time.sleep(0.1)
                continue
            
            # Run detection with live_feed_detector (ONLY distances, NO counting)
            annotated_rgb, count, error, bundle_info = live_feed_detector.detect_rebars(
                frame, model, depth_map=depth_map, conf=0.5, iou=0.3, max_det=10000
            )
            
            if bundle_info:
                bundles = bundle_info.get("bundles", [])
                bundles_with_distances = []
                for bundle in bundles:
                    if bundle.get("distance_m") is not None:
                        bundles_with_distances.append(bundle)
                
                latest_detection_result = {
                    "bundles": bundles_with_distances,
                    "nearest_bundle": bundle_info.get("nearest_bundle"),
                    "total_bundles": len(bundles_with_distances),
                    "timestamp": time.time(),
                    "annotated_frame": annotated_rgb
                }
            
            time.sleep(detection_interval)
            
        except Exception as e:
            print(f"[Continuous Detection] Error: {e}")
            time.sleep(1)
    
    print("[Continuous Detection] Stopped")


def oak_mjpeg_generator_with_distances_only():
    """
    Generator for MJPEG streaming showing ONLY bundle distances (NO counting, NO rebar IDs)
    """
    frame_count = 0
    fps_buffer = deque(maxlen=30)
    
    try:
        while True:
            start_time = time.time()
            
            # Get raw frame from OAK-D
            frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=0.1)
            
            if err or frame is None:
                # Show error frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "OAK-D Camera Error",
                    (150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                if err:
                    error_text = str(err)[:40]
                    cv2.putText(
                        placeholder,
                        error_text,
                        (100, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                ok, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    jpg = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                time.sleep(0.5)
                continue
            
            # Start with raw frame
            display_frame = frame.copy()
            
            # Draw bundle information (ONLY rectangles and distances, NO IDs, NO counts)
            if latest_detection_result.get("bundles"):
                display_frame = draw_bundle_distances_only(display_frame, latest_detection_result)
            
            # Add FPS counter
            frame_time = time.time() - start_time
            fps_buffer.append(1.0 / max(frame_time, 0.001))
            if len(fps_buffer) > 0:
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                cv2.putText(
                    display_frame,
                    f"FPS: {avg_fps:.1f}",
                    (display_frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                display_frame,
                timestamp,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            
            # Add instruction text
            cv2.putText(
                display_frame,
                "Press CAPTURE to count bundles",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            
            # Encode to JPEG
            ok, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                time.sleep(0.05)
                continue
            
            jpg = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            
            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # ~30 FPS max
                time.sleep(0.033 - elapsed)
            
    except GeneratorExit:
        pass
    except Exception as e:
        print(f"[Stream] Error: {e}")
    finally:
        pass


def draw_bundle_distances_only(frame, detection_result):
    """
    Draw ONLY bundle rectangles and distances on live feed - NO counting, NO rebar IDs
    """
    if frame is None:
        return frame
    
    frame = frame.copy()
    bundles = detection_result.get("bundles", [])
    nearest_bundle = detection_result.get("nearest_bundle")
    
    if not bundles:
        return frame
    
    # Colors for different bundles
    colors = [
        (0, 255, 0),    # Green - nearest
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (0, 165, 255),  # Orange
    ]
    
    # Draw all bundles with distance text
    for idx, bundle in enumerate(bundles):
        bounds = bundle.get("bounds")
        distance = bundle.get("distance_m")
        bundle_id = bundle.get("bundle_id")
        
        if bounds and distance:
            x1, y1, x2, y2 = map(int, bounds)
            
            # Determine color
            is_nearest = (nearest_bundle and bundle_id == nearest_bundle.get("bundle_id"))
            color = colors[0] if is_nearest else colors[(idx + 1) % len(colors)]
            thickness = 3 if is_nearest else 2
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw distance text above rectangle
            distance_text = f"{distance:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(distance_text, font, font_scale, thickness)
            
            # Position text above the rectangle
            text_x = x1 + (x2 - x1)//2 - text_w//2
            text_y = y1 - 10
            
            # Ensure text is within frame
            if text_y < text_h + 10:
                text_y = y2 + text_h + 10
            
            # Draw background for text
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_h - 5),
                (text_x + text_w + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw distance text
            cv2.putText(
                frame,
                distance_text,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            
            # Draw bundle ID
            id_text = f"B{bundle_id}"
            cv2.putText(
                frame,
                id_text,
                (x1 + 5, y1 + 20),
                font,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
            
            # For nearest bundle, draw "NEAREST" label
            if is_nearest:
                nearest_text = "NEAREST"
                cv2.putText(
                    frame,
                    nearest_text,
                    (x1, y1 - 5),
                    font,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
    
    # Add info panel at bottom
    h, w = frame.shape[:2]
    panel_height = 80
    panel_y = h - panel_height
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Show nearest bundle info
    if nearest_bundle and nearest_bundle.get("distance_m"):
        cv2.putText(
            frame,
            f"NEAREST BUNDLE: {nearest_bundle['distance_m']:.2f}m",
            (20, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Total Bundles: {len(bundles)}",
            (20, panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    else:
        cv2.putText(
            frame,
            "No bundles detected. Move camera closer to rebar bundles.",
            (20, panel_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    
    return frame


@simple_router.post("/capture-and-count")
def capture_and_count(user_id: int = Form(...)):
    """
    Capture and count bundles with smart logic:
    - If distance difference between ANY bundles > 0.20m: Count ONLY nearest bundle
    - If distance difference between ALL consecutive bundles ≤ 0.20m: Count ALL bundles separately
    Returns annotated image with ONLY counted bundles' rebars having unique IDs
    Saves the detection to the database
    """
    print(f"\n{'='*50}")
    print(f"[Capture] Request for user {user_id}")
    
    # Get current frame
    frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=1.0)
    
    if err or frame is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": err or "Failed to capture frame",
                "total_rebars": 0,
                "bundles_counted": []
            }
        )
    
    # Get ALL bundles (no filtering yet)
    annotated_rgb, total_rebars, error, bundle_info = capture_detector.detect_rebars(
        frame, model, depth_map=depth_map, conf=0.5, iou=0.3, max_det=10000
    )
    
    if error:
        return JSONResponse(
            status_code=500,
            content={
                "error": error,
                "total_rebars": 0,
                "bundles_counted": []
            }
        )
    
    bundles = bundle_info.get("bundles", []) if bundle_info else []
    
    # Filter bundles with valid distances
    valid_bundles = [b for b in bundles if b.get("distance_m") is not None]
    
    if not valid_bundles:
        return {
            "total_rebars": 0,
            "counting_mode": "none",
            "bundles_counted": [],
            "all_bundles": [],
            "error": "No bundles with valid distances detected",
            "image": None
        }
    
    # Sort bundles by distance
    sorted_bundles = sorted(valid_bundles, key=lambda b: b["distance_m"])
    
    # Check distance differences between consecutive bundles
    distance_diffs = []
    max_diff = 0
    for i in range(len(sorted_bundles) - 1):
        diff = abs(sorted_bundles[i]["distance_m"] - sorted_bundles[i+1]["distance_m"])
        distance_diffs.append({
            "bundle1_id": sorted_bundles[i]["bundle_id"],
            "bundle1_distance": sorted_bundles[i]["distance_m"],
            "bundle2_id": sorted_bundles[i+1]["bundle_id"],
            "bundle2_distance": sorted_bundles[i+1]["distance_m"],
            "difference": round(diff, 2)
        })
        if diff > max_diff:
            max_diff = diff
    
    # Determine which bundles to count
    counted_bundle_ids = set()
    bundles_counted = []
    total_count = 0
    
    if max_diff > 0.20:
        # Count ONLY the nearest bundle
        print(f"[Capture] Max distance difference = {max_diff:.2f}m > 0.20m")
        print(f"[Capture] Counting ONLY nearest bundle")
        
        nearest = sorted_bundles[0]
        counted_bundle_ids.add(nearest["bundle_id"])
        bundles_counted.append({
            "bundle_id": nearest["bundle_id"],
            "distance_m": nearest["distance_m"],
            "rebar_count": nearest["size"],
            "display_text": f"Bundle {nearest['bundle_id']} = {nearest['size']} bars",
            "counted": True
        })
        total_count = nearest["size"]
        counting_mode = "nearest_only"
        
        print(f"  ✓ Bundle {nearest['bundle_id']}: {nearest['size']} bars at {nearest['distance_m']:.2f}m (COUNTED)")
        for bundle in sorted_bundles[1:]:
            print(f"  ✗ Bundle {bundle['bundle_id']}: {bundle['size']} bars at {bundle['distance_m']:.2f}m (IGNORED)")
        
    else:
        # Count ALL bundles
        print(f"[Capture] Max distance difference = {max_diff:.2f}m ≤ 0.20m")
        print(f"[Capture] Counting ALL bundles separately")
        
        for bundle in sorted_bundles:
            counted_bundle_ids.add(bundle["bundle_id"])
            bundles_counted.append({
                "bundle_id": bundle["bundle_id"],
                "distance_m": bundle["distance_m"],
                "rebar_count": bundle["size"],
                "display_text": f"Bundle {bundle['bundle_id']} = {bundle['size']} bars",
                "counted": True
            })
            total_count += bundle["size"]
            print(f"  ✓ Bundle {bundle['bundle_id']}: {bundle['size']} bars at {bundle['distance_m']:.2f}m")
        
        counting_mode = "all_bundles_separately"
    
    # Create bundles detail for frontend
    bundles_detail = []
    for bundle in bundles_counted:
        bundles_detail.append({
            "bundle_id": bundle["bundle_id"],
            "rebar_count": bundle["rebar_count"],
            "distance_m": bundle["distance_m"],
            "counted": bundle["counted"]
        })
    
    # Extract all boxes from the detection
    all_boxes = []
    for bundle in bundles:
        for rebar in bundle.get("rebars", []):
            all_boxes.append(rebar["box"])
    
    # Create clean annotated image with ONLY counted bundles' rebars
    if all_boxes:
        boxes_array = np.array(all_boxes, dtype=np.float32)
        annotated_result = capture_detector.annotate_counted_bundles(
            frame, boxes_array, bundle_info, counted_bundle_ids
        )
    else:
        annotated_result = frame.copy()
    
    # Convert to RGB for display and saving
    annotated_rgb_result = cv2.cvtColor(annotated_result, cv2.COLOR_BGR2RGB)
    
    # SAVE TO DATABASE
    det_id = None
    try:
        det_id = record_detection(
            user_id=user_id,
            processed_rgb=annotated_rgb_result,
            count=total_count,
            stream_url="OAK-D Pro Capture",
            snapshot_url="Live Capture",
            bundle_info=bundle_info,
        )
        print(f"[Capture] ✅ Saved to database with ID: {det_id}")
    except Exception as e:
        print(f"[Capture] ❌ Error saving to database: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert to data URI for frontend display
    image_uri = None
    if annotated_result is not None:
        image_uri = img_to_data_uri(annotated_rgb_result, quality=88, max_w=720)
    
    # Create display summary
    if counting_mode == "nearest_only":
        if bundles_counted:
            display_summary = f"Nearest Bundle: Bundle {bundles_counted[0]['bundle_id']} = {bundles_counted[0]['rebar_count']} bars"
        else:
            display_summary = "No bundles counted"
    else:
        display_summary = " | ".join([f"Bundle {b['bundle_id']} = {b['rebar_count']} bars" for b in bundles_counted])
    
    response = {
        "det_id": det_id,
        "total_rebars": total_count,
        "counting_mode": counting_mode,
        "bundles_counted": bundles_counted,
        "bundles_detail": bundles_detail,
        "total_bundles": len([b for b in bundles_counted if b["counted"]]),
        "all_bundles": [{
            "bundle_id": b["bundle_id"],
            "distance_m": b["distance_m"],
            "rebar_count": b["size"],
            "counted": b["bundle_id"] in counted_bundle_ids
        } for b in sorted_bundles],
        "distance_differences": distance_diffs,
        "max_distance_difference": round(max_diff, 2),
        "display_summary": display_summary,
        "error": None,
        "image": image_uri,
    }
    
    print(f"\n[Capture Result]")
    print(f"  Mode: {counting_mode}")
    print(f"  Max difference: {max_diff:.2f}m")
    print(f"  Total counted: {total_count} rebars")
    print(f"  Total bundles counted: {len(bundles_counted)}")
    print(f"  Bundles: {[b['bundle_id'] for b in bundles_counted]}")
    print(f"  Saved to DB: {'✅ Yes' if det_id else '❌ No'}")
    print(f"{'='*50}\n")
    
    return response


@simple_router.get("/oak-stream")
def oak_stream():
    """
    OAK-D live preview stream showing bundle distances.
    Press capture button to count bundles.
    """
    return StreamingResponse(
        oak_mjpeg_generator_with_distances_only(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@simple_router.get("/oak-snapshot")
def oak_snapshot():
    """
    Returns a single raw JPEG frame from the OAK-D camera.
    """
    frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=1.0)
    
    if err or frame is None:
        return Response(status_code=503, content=b"Camera error")

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return Response(status_code=500, content=b"Encoding error")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@simple_router.get("/live-bundle-info")
def get_live_bundle_info():
    """
    Get current bundle distance information for UI display.
    """
    return {
        "bundles": latest_detection_result.get("bundles", []),
        "nearest_bundle": latest_detection_result.get("nearest_bundle"),
        "total_bundles": latest_detection_result.get("total_bundles", 0),
        "timestamp": latest_detection_result.get("timestamp", 0)
    }


# Legacy endpoints (for compatibility)
@simple_router.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    result = capture_detector.detect_image(contents)
    return result


@simple_router.post("/oak-d")
def detect_oak_d():
    frame, depth_map, grab_error = grab_oak_frame(OAK_SESSION, wait_sec=2.0)
    
    if grab_error or frame is None:
        return {
            "count": 0,
            "error": grab_error or "Failed to grab frame from OAK-D Pro.",
            "bundle_info": None,
            "image": None,
        }

    result = capture_detector.detect_oak_camera(frame, depth_map=depth_map)
    return result


# Attach routers
app.include_router(simple_router)
app.include_router(auth_routes.router)
app.include_router(detection_routes.router)


@app.on_event("startup")
def on_startup():
    """
    Initialize database and start background detection thread.
    """
    init_db()
    
    # Start background detection thread
    global detection_thread
    detection_thread = threading.Thread(target=continuous_detection, daemon=True)
    detection_thread.start()
    
    print("✅ Rebar-Counting backend started with:")
    print("   - Live feed: Shows distances for all bundles")
    print("   - Capture: Only counted bundles get unique IDs")
    print("   - Ignored bundles: No annotations at all")
    print("   - Captures are saved to database")
    print("   - Background detection running every 0.5s")


@app.on_event("shutdown")
def on_shutdown():
    """
    Stop background thread and close OAK-D device.
    """
    global detection_thread_running
    detection_thread_running = False
    time.sleep(1)
    
    close_oak_device(OAK_SESSION)
    print("✅ OAK-D device closed")


@app.get("/")
def root():
    return {
        "message": "Rebar-Counting backend is running",
        "mode": "Live feed shows distances - Capture shows unique IDs only for counted bundles",
        "distance_tolerance_m": 0.20,
        "endpoints": {
            "live_stream": "/oak-stream",
            "capture_count": "/capture-and-count",
            "live_info": "/live-bundle-info",
            "snapshot": "/oak-snapshot"
        }
    }
