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

# Global OAK session + lock (IMPORTANT)
OAK_SESSION = {}
OAK_LOCK = threading.Lock()

# ---------------------------
# Tuning (start values)
# ---------------------------
LIVE_CONF = 0.25
LIVE_IOU = 0.7

CAPTURE_CONF = 0.25
CAPTURE_IOU = 0.7

# Crop-zoom for very small rebars (recommended)
CAPTURE_USE_CROP_ZOOM = True
CAPTURE_ZOOM_MARGIN_PX = 80  # expand around predicted bundle region
CAPTURE_ZOOM_MIN_SIZE = 320  # minimum crop width/height

# Latest detection cache used by live stream overlay + capture ROI suggestion
latest_detection_result = {
    "bundles": [],
    "nearest_bundle": None,
    "timestamp": 0.0,
    "sorted_bundles": [],
    "counted_bundle_ids": set(),
    "bundle_rebar_ranges": {},
    "total_rebars": 0,
    "counting_mode": "none",
    "total_bundles": 0,
    "raw_frame": None,
}

# Detector for live feed (bundles + distances, no per-rebar labels in stream)
live_feed_detector = RebarBundleDetector(
    min_bundle_size=3,
    min_samples=2,
    row_tolerance=40.0,
    eps_scale=1.25,
    eps_min=10.0,
    eps_max=500.0,
    use_depth_filter=True,
    max_detection_distance_mm=1500.0,
    min_detection_distance_mm=200.0,
    track_bundle_distances=True,
    nearest_bundle_only=False,
    draw_seg_masks=False,
    debug=False,
)

# Detector for capture (counting + optional seg masks)
capture_detector = RebarBundleDetector(
    min_bundle_size=3,
    min_samples=2,
    row_tolerance=40.0,
    eps_scale=1.25,
    eps_min=10.0,
    eps_max=500.0,
    use_depth_filter=True,
    max_detection_distance_mm=1500.0,
    min_detection_distance_mm=200.0,
    track_bundle_distances=True,
    nearest_bundle_only=False,
    draw_seg_masks=True,
    debug=False,
)

simple_router = APIRouter(tags=["simple-detection"])

# Background thread for continuous detection on live feed
detection_thread_running = True
detection_interval = 0.5


# ---------------------------
# Helpers
# ---------------------------
def _clip_roi(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def _choose_capture_roi_from_latest(frame_shape):
    """
    Use latest live detection bundle bounds as ROI hint for crop-zoom.
    Returns (x1,y1,x2,y2) or None.
    """
    det = latest_detection_result or {}
    sorted_bundles = det.get("sorted_bundles") or []
    if not sorted_bundles:
        return None

    # union bounds of all bundles we currently see
    xs1, ys1, xs2, ys2 = [], [], [], []
    for b in sorted_bundles:
        bounds = b.get("bounds")
        if not bounds:
            continue
        x1, y1, x2, y2 = map(int, bounds)
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)

    if not xs1:
        return None

    h, w = frame_shape[:2]
    x1 = min(xs1) - CAPTURE_ZOOM_MARGIN_PX
    y1 = min(ys1) - CAPTURE_ZOOM_MARGIN_PX
    x2 = max(xs2) + CAPTURE_ZOOM_MARGIN_PX
    y2 = max(ys2) + CAPTURE_ZOOM_MARGIN_PX

    # ensure minimum crop size
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    crop_w = max(CAPTURE_ZOOM_MIN_SIZE, x2 - x1)
    crop_h = max(CAPTURE_ZOOM_MIN_SIZE, y2 - y1)

    x1 = cx - crop_w // 2
    x2 = cx + crop_w // 2
    y1 = cy - crop_h // 2
    y2 = cy + crop_h // 2

    return _clip_roi(x1, y1, x2, y2, w, h)


def _shift_bundle_info(bundle_info, dx, dy):
    """
    Shift bundle_info boxes/bounds/centers from crop coords to full-image coords.
    """
    if not bundle_info:
        return bundle_info

    for b in bundle_info.get("bundles", []) or []:
        if b.get("bounds"):
            x1, y1, x2, y2 = b["bounds"]
            b["bounds"] = [float(x1 + dx), float(y1 + dy), float(x2 + dx), float(y2 + dy)]
        for r in b.get("rebars", []) or []:
            if r.get("box"):
                x1, y1, x2, y2 = r["box"]
                r["box"] = [float(x1 + dx), float(y1 + dy), float(x2 + dx), float(y2 + dy)]
            if r.get("center"):
                cx, cy = r["center"]
                r["center"] = [float(cx + dx), float(cy + dy)]
    return bundle_info


def _compute_ranges_and_total(sorted_bundles, counted_bundle_ids):
    bundle_rebar_ranges = {}
    global_counter = 1
    total_rebars = 0
    for b in sorted_bundles:
        bid = b.get("bundle_id")
        if bid in counted_bundle_ids:
            n = int(b.get("size", 0) or 0)
            if n > 0:
                bundle_rebar_ranges[bid] = (global_counter, global_counter + n - 1)
                global_counter += n
                total_rebars += n
    return bundle_rebar_ranges, total_rebars


def _detect_with_depth_fallback(frame_bgr, depth_map):
    """
    Try with depth first (distances + depth filtering).
    If it produces no bundles, retry RGB-only (depth_map=None).
    Returns: (annotated_rgb, count, error, bundle_info, depth_used)
    """
    annotated_rgb, count, error, bundle_info = capture_detector.detect_rebars(
        frame_bgr, model, depth_map=depth_map, conf=CAPTURE_CONF, iou=CAPTURE_IOU, max_det=10000
    )
    bundles = (bundle_info or {}).get("bundles") if bundle_info else None
    if (error is None) and bundles and len(bundles) > 0:
        return annotated_rgb, count, None, bundle_info, True

    # Fallback: RGB-only (still runs model + masks, but no distances)
    annotated_rgb2, count2, error2, bundle_info2 = capture_detector.detect_rebars(
        frame_bgr, model, depth_map=None, conf=CAPTURE_CONF, iou=CAPTURE_IOU, max_det=10000
    )
    return annotated_rgb2, count2, error2, bundle_info2, False


def _compute_counting_from_bundles(bundles):
    """
    Smart counting mode:
      - if distances exist and gap >0.20m => nearest_only
      - else count all
      - if no distances => count all (no_depth)
    Returns: (sorted_bundles, counted_bundle_ids, counting_mode, max_diff, diffs)
    """
    if not bundles:
        return [], set(), "none", 0.0, []

    valid = [b for b in bundles if b.get("distance_m") is not None]

    if valid:
        sorted_bundles = sorted(valid, key=lambda b: b["distance_m"])
        diffs = []
        max_diff = 0.0
        if len(sorted_bundles) > 1:
            for i in range(len(sorted_bundles) - 1):
                d1 = sorted_bundles[i].get("distance_m")
                d2 = sorted_bundles[i + 1].get("distance_m")
                if d1 is None or d2 is None:
                    continue
                diff = abs(float(d1) - float(d2))
                max_diff = max(max_diff, diff)
                diffs.append(
                    {
                        "bundle1_id": sorted_bundles[i]["bundle_id"],
                        "bundle1_distance": d1,
                        "bundle2_id": sorted_bundles[i + 1]["bundle_id"],
                        "bundle2_distance": d2,
                        "difference": round(diff, 2),
                    }
                )

        if max_diff > 0.20:
            counting_mode = "nearest_only"
            counted_bundle_ids = {sorted_bundles[0]["bundle_id"]}
        else:
            counting_mode = "all_bundles_separately"
            counted_bundle_ids = {b["bundle_id"] for b in sorted_bundles}
        return sorted_bundles, counted_bundle_ids, counting_mode, max_diff, diffs

    # No depth distances -> still count
    sorted_bundles = bundles
    counting_mode = "all_bundles_no_depth"
    counted_bundle_ids = {b["bundle_id"] for b in sorted_bundles if b.get("bundle_id") is not None}
    return sorted_bundles, counted_bundle_ids, counting_mode, 0.0, []


# ---------------------------
# Background continuous detection (live cache)
# ---------------------------
def _compute_live_result(frame, depth_map):
    """
    Live result should NOT error if distances missing.
    We keep bundles even without depth, but live drawing will show distance only if available.
    """
    annotated_rgb, count, error, bundle_info, depth_used = _detect_with_depth_fallback(frame, depth_map)
    if error or not bundle_info:
        return None

    bundles = bundle_info.get("bundles", []) or []
    if not bundles:
        return None

    sorted_bundles, counted_bundle_ids, counting_mode, max_diff, diffs = _compute_counting_from_bundles(bundles)
    bundle_rebar_ranges, total_rebars = _compute_ranges_and_total(sorted_bundles, counted_bundle_ids)

    return {
        "bundles": bundles,
        "nearest_bundle": bundle_info.get("nearest_bundle"),
        "total_bundles": len(bundles),
        "sorted_bundles": sorted_bundles,
        "counted_bundle_ids": counted_bundle_ids,
        "bundle_rebar_ranges": bundle_rebar_ranges,
        "total_rebars": total_rebars,
        "counting_mode": counting_mode,
        "timestamp": time.time(),
        "raw_frame": frame,
    }


def continuous_detection():
    global latest_detection_result, detection_thread_running

    print("[Continuous Detection] Started")

    while detection_thread_running:
        try:
            with OAK_LOCK:
                frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=0.1)

            if err or frame is None:
                time.sleep(0.1)
                continue

            result = _compute_live_result(frame, depth_map)
            if result:
                latest_detection_result = result

            time.sleep(detection_interval)

        except Exception as e:
            print(f"[Continuous Detection] Error: {e}")
            time.sleep(1)

    print("[Continuous Detection] Stopped")


# ---------------------------
# Live drawing
# ---------------------------
def draw_live_bundles_only(frame, sorted_bundles):
    """
    Draw ONLY bundle rectangles and labels (B1, B2, …) and distances if available.
    """
    if frame is None or not sorted_bundles:
        return frame

    display = frame.copy()
    h, w = display.shape[:2]

    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, bundle in enumerate(sorted_bundles):
        bounds = bundle.get("bounds")
        if not bounds:
            continue

        x1, y1, x2, y2 = map(int, bounds)
        color = colors[idx % len(colors)]

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        label = f"B{idx + 1}"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.65, 2)
        lx = min(x2 + 6, w - lw - 4)
        ly = y1 + lh
        cv2.rectangle(display, (lx - 3, ly - lh - 3), (lx + lw + 3, ly + 3), (0, 0, 0), -1)
        cv2.putText(display, label, (lx, ly), font, 0.65, color, 2, cv2.LINE_AA)

        dist = bundle.get("distance_m")
        if dist is not None:
            txt = f"{float(dist):.2f}m"
            (tw, th), _ = cv2.getTextSize(txt, font, 0.6, 2)
            tx = x1 + (x2 - x1) // 2 - tw // 2
            ty = y1 - 10
            if ty < th + 10:
                ty = y2 + th + 10
            cv2.rectangle(display, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
            cv2.putText(display, txt, (tx, ty), font, 0.6, color, 2, cv2.LINE_AA)

    # bottom bar
    panel_height = 50
    panel_y = h - panel_height
    overlay = display.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
    display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

    cv2.putText(
        display,
        f"Bundles detected: {len(sorted_bundles)}",
        (20, panel_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return display


def oak_mjpeg_generator_with_distances_only():
    fps_buffer = deque(maxlen=30)

    try:
        while True:
            start_time = time.time()

            with OAK_LOCK:
                frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=0.1)

            if err or frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "OAK-D Camera Error", (150, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ok, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                time.sleep(0.5)
                continue

            display_frame = frame.copy()
            det = latest_detection_result or {}
            sorted_bundles = det.get("sorted_bundles", []) or []
            if sorted_bundles:
                display_frame = draw_live_bundles_only(display_frame, sorted_bundles)

            frame_time = time.time() - start_time
            fps_buffer.append(1.0 / max(frame_time, 0.001))
            avg_fps = sum(fps_buffer) / len(fps_buffer)

            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                        (display_frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display_frame, time.strftime("%Y-%m-%d %H:%M:%S"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            ok, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                time.sleep(0.05)
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

            elapsed = time.time() - start_time
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    except GeneratorExit:
        pass
    except Exception as e:
        print(f"[Stream] Error: {e}")


# ---------------------------
# Capture drawing (your existing behavior)
# ---------------------------
def draw_captured_bundles(frame, sorted_bundles, counted_bundle_ids, bundle_rebar_ranges=None):
    if frame is None:
        return frame

    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    bundle_colors = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
    ]
    uncounted_color = (100, 100, 100)

    for bundle_idx, bundle in enumerate(sorted_bundles):
        bundle_id = bundle.get("bundle_id")
        bounds = bundle.get("bounds")
        rebars = bundle.get("rebars", [])
        distance = bundle.get("distance_m")

        if not bounds:
            continue

        x1, y1, x2, y2 = map(int, bounds)
        is_counted = bundle_id in counted_bundle_ids
        color = bundle_colors[bundle_idx % len(bundle_colors)] if is_counted else uncounted_color
        bundle_label = f"B{bundle_idx + 1}"

        rect_thickness = 3 if is_counted else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, rect_thickness)

        (lw, lh), _ = cv2.getTextSize(bundle_label, font, 0.75, 2)
        lx = min(x2 + 6, frame.shape[1] - lw - 4)
        ly = y1 + lh
        cv2.rectangle(frame, (lx - 3, ly - lh - 3), (lx + lw + 3, ly + 3), (0, 0, 0), -1)
        cv2.putText(frame, bundle_label, (lx, ly), font, 0.75, color, 2, cv2.LINE_AA)

        if distance is not None:
            dist_text = f"{float(distance):.2f}m"
            (tw, th), _ = cv2.getTextSize(dist_text, font, 0.55, 1)
            tx = x1 + (x2 - x1) // 2 - tw // 2
            ty = y1 - 10
            if ty < th + 10:
                ty = y2 + th + 10
            cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(frame, dist_text, (tx, ty), font, 0.55, color, 1, cv2.LINE_AA)

        if not is_counted:
            cv2.putText(frame, "IGNORED", (x1 + 6, y1 + 22),
                        font, 0.55, uncounted_color, 1, cv2.LINE_AA)
            continue

        bundle_range = bundle_rebar_ranges.get(bundle_id) if bundle_rebar_ranges else None

        for rebar_local_idx, rebar in enumerate(rebars):
            box = rebar.get("box")
            if box is None:
                continue

            rx1, ry1, rx2, ry2 = map(int, box)

            if bundle_range is not None:
                rebar_num = int(bundle_range[0]) + rebar_local_idx
            else:
                rebar_num = rebar_local_idx + 1

            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            cx = (rx1 + rx2) // 2
            cy = (ry1 + ry2) // 2

            num_text = str(rebar_num)
            font_scale = 0.35
            thickness = 1
            (nw, nh), _ = cv2.getTextSize(num_text, font, font_scale, thickness)
            tx = cx - nw // 2
            ty = cy + nh // 2

            cv2.rectangle(frame, (tx - 2, ty - nh - 2), (tx + nw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(frame, num_text, (tx, ty), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


# ---------------------------
# Capture endpoint
# ---------------------------
@simple_router.post("/capture-and-count")
def capture_and_count(user_id: int = Form(...)):
    print(f"\n{'='*50}")
    print(f"[Capture] Request for user {user_id}")

    # Grab a fresh frame
    with OAK_LOCK:
        frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=1.0)

    if err or frame is None:
        return JSONResponse(
            status_code=503,
            content={"error": err or "Failed to capture frame", "total_rebars": 0, "bundles_counted": []},
        )

    # Crop-zoom around the bundle area (improves tiny rebar detection)
    roi = _choose_capture_roi_from_latest(frame.shape) if CAPTURE_USE_CROP_ZOOM else None
    used_roi = False

    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        depth_crop = depth_map[y1:y2, x1:x2] if depth_map is not None else None

        annotated_rgb_crop, count, error, bundle_info, depth_used = _detect_with_depth_fallback(crop, depth_crop)

        # If we detected bundles in ROI, use it
        bundles = (bundle_info or {}).get("bundles") if bundle_info else []
        if (error is None) and bundles:
            used_roi = True

            # shift bundle_info to full-frame coords
            bundle_info = _shift_bundle_info(bundle_info, dx=x1, dy=y1)

            # Build full annotated RGB by pasting ROI annotated region
            full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_rgb[y1:y2, x1:x2] = annotated_rgb_crop
            annotated_rgb = full_rgb
        else:
            # fallback to full-frame detection
            annotated_rgb, count, error, bundle_info, depth_used = _detect_with_depth_fallback(frame, depth_map)
    else:
        annotated_rgb, count, error, bundle_info, depth_used = _detect_with_depth_fallback(frame, depth_map)

    if error:
        return JSONResponse(
            status_code=500,
            content={"error": error, "total_rebars": 0, "bundles_counted": []},
        )

    bundles = bundle_info.get("bundles", []) if bundle_info else []
    if not bundles:
        return {
            "det_id": None,
            "total_rebars": 0,
            "counting_mode": "none",
            "bundles_counted": [],
            "bundles_detail": [],
            "total_bundles": 0,
            "all_bundles": [],
            "distance_differences": [],
            "max_distance_difference": 0.0,
            "display_summary": "No bundles detected",
            "error": "No bundles detected (even RGB-only fallback).",
            "image": img_to_data_uri(annotated_rgb, quality=88, max_w=720) if isinstance(annotated_rgb, np.ndarray) else None,
        }

    # Decide sorting/counting mode (DO NOT require distance anymore)
    sorted_bundles, counted_bundle_ids, counting_mode, max_diff, distance_diffs = _compute_counting_from_bundles(bundles)
    bundle_rebar_ranges, total_count = _compute_ranges_and_total(sorted_bundles, counted_bundle_ids)

    # Build bundles_detail
    bundles_detail = []
    for bi, b in enumerate(sorted_bundles, start=1):
        bid = b.get("bundle_id")
        bundles_detail.append(
            {
                "bundle_id": bid,
                "bundle_label": f"B{bi}",
                "rebar_count": int(b.get("size", 0) or 0),
                "distance_m": b.get("distance_m"),
                "counted": bid in counted_bundle_ids,
            }
        )

    # Draw capture result on top of model output (with seg masks if model produced them)
    output_bgr = frame
    if isinstance(annotated_rgb, np.ndarray):
        try:
            output_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            output_bgr = frame

    annotated_bgr = draw_captured_bundles(
        output_bgr, sorted_bundles, counted_bundle_ids, bundle_rebar_ranges=bundle_rebar_ranges
    )
    annotated_rgb_result = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Save to DB
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
    except Exception as e:
        print(f"[Capture] ❌ Error saving to database: {e}")

    # Display summary
    display_summary = f"Total = {total_count} bars"

    return {
        "det_id": det_id,
        "total_rebars": total_count,
        "counting_mode": counting_mode,
        "bundles_counted": [
            {
                "bundle_id": b["bundle_id"],
                "bundle_label": b.get("bundle_label"),
                "distance_m": b.get("distance_m"),
                "rebar_count": b.get("rebar_count"),
                "display_text": f"{b.get('bundle_label')} = {b.get('rebar_count')} bars",
                "counted": b.get("counted", True),
            }
            for b in bundles_detail
            if b.get("counted")
        ],
        "bundles_detail": bundles_detail,
        "total_bundles": len(sorted_bundles),
        "all_bundles": bundles_detail,
        "distance_differences": distance_diffs,
        "max_distance_difference": round(float(max_diff), 2),
        "display_summary": display_summary,
        "error": None,
        "image": img_to_data_uri(annotated_rgb_result, quality=88, max_w=720),
        "debug": {
            "depth_used": bool(depth_used),
            "used_roi_zoom": bool(used_roi),
        },
    }


# ---------------------------
# Other endpoints
# ---------------------------
@simple_router.get("/oak-stream")
def oak_stream():
    return StreamingResponse(
        oak_mjpeg_generator_with_distances_only(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@simple_router.get("/oak-snapshot")
def oak_snapshot():
    with OAK_LOCK:
        frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=1.0)

    if err or frame is None:
        return Response(status_code=503, content=b"Camera error")

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return Response(status_code=500, content=b"Encoding error")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@simple_router.get("/live-bundle-info")
def get_live_bundle_info():
    return {
        "bundles": latest_detection_result.get("bundles", []),
        "nearest_bundle": latest_detection_result.get("nearest_bundle"),
        "total_bundles": latest_detection_result.get("total_bundles", 0),
        "timestamp": latest_detection_result.get("timestamp", 0),
    }


# Legacy endpoints
@simple_router.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    result = capture_detector.detect_image(contents)
    return result


@simple_router.post("/oak-d")
def detect_oak_d():
    with OAK_LOCK:
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
    init_db()

    global detection_thread
    detection_thread = threading.Thread(target=continuous_detection, daemon=True)
    detection_thread.start()

    print("✅ Rebar-Counting backend started")
    print("   - OAK_LOCK enabled")
    print("   - Depth fallback enabled (RGB-only retry)")
    print(f"   - Capture crop-zoom: {CAPTURE_USE_CROP_ZOOM}")


@app.on_event("shutdown")
def on_shutdown():
    global detection_thread_running
    detection_thread_running = False
    time.sleep(1)

    close_oak_device(OAK_SESSION)
    print("✅ OAK-D device closed")


@app.get("/")
def root():
    return {
        "message": "Rebar-Counting backend is running",
        "distance_tolerance_m": 0.20,
        "endpoints": {
            "live_stream": "/oak-stream",
            "capture_count": "/capture-and-count",
            "live_info": "/live-bundle-info",
            "snapshot": "/oak-snapshot",
        },
    }