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

# Latest detection result cache — full counting result for live display
latest_detection_result = {
    "bundles": [],
    "nearest_bundle": None,
    "timestamp": 0,
    "sorted_bundles": [],
    "counted_bundle_ids": set(),
    "bundle_rebar_ranges": {},
    "total_rebars": 0,
    "counting_mode": "none",
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


def _compute_counting_result(frame, depth_map):
    """
    Run full bundle detection + counting logic on a frame.
    Returns a dict ready to be stored in latest_detection_result.
    """
    _, _total, error, bundle_info = capture_detector.detect_rebars(
        frame, model, depth_map=depth_map, conf=0.5, iou=0.3, max_det=10000
    )
    if error or not bundle_info:
        return None

    bundles = bundle_info.get("bundles", [])
    valid_bundles = [b for b in bundles if b.get("distance_m") is not None]
    if not valid_bundles:
        return None

    sorted_bundles = sorted(valid_bundles, key=lambda b: b["distance_m"])

    max_diff = 0.0
    for i in range(len(sorted_bundles) - 1):
        diff = abs(sorted_bundles[i]["distance_m"] - sorted_bundles[i + 1]["distance_m"])
        if diff > max_diff:
            max_diff = diff

    if max_diff > 0.20:
        counting_mode = "nearest_only"
        counted_bundle_ids = {sorted_bundles[0]["bundle_id"]}
    else:
        counting_mode = "all_bundles_separately"
        counted_bundle_ids = {b["bundle_id"] for b in sorted_bundles}

    bundle_rebar_ranges = {}
    global_counter = 1
    total_rebars = 0
    for bundle in sorted_bundles:
        bid = bundle["bundle_id"]
        if bid in counted_bundle_ids:
            n = bundle["size"]
            bundle_rebar_ranges[bid] = (global_counter, global_counter + n - 1)
            global_counter += n
            total_rebars += n

    return {
        "bundles": valid_bundles,
        "nearest_bundle": bundle_info.get("nearest_bundle"),
        "total_bundles": len(valid_bundles),
        "sorted_bundles": sorted_bundles,
        "counted_bundle_ids": counted_bundle_ids,
        "bundle_rebar_ranges": bundle_rebar_ranges,
        "total_rebars": total_rebars,
        "counting_mode": counting_mode,
        "timestamp": time.time(),
        "raw_frame": frame,
    }


def continuous_detection():
    """Background thread: runs full counting on every frame so the live stream
    always shows B-labels, rebar numbers, and counts — no manual capture needed."""
    global latest_detection_result, detection_thread_running

    print("[Continuous Detection] Started - Full auto-counting on every frame")

    while detection_thread_running:
        try:
            frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=0.1)
            if err or frame is None:
                time.sleep(0.1)
                continue

            result = _compute_counting_result(frame, depth_map)
            if result:
                latest_detection_result = result

            time.sleep(detection_interval)

        except Exception as e:
            print(f"[Continuous Detection] Error: {e}")
            time.sleep(1)

    print("[Continuous Detection] Stopped")


def oak_mjpeg_generator_with_distances_only():
    """
    MJPEG stream that automatically shows full counting results on every frame —
    B1/B2 labels, sequential rebar numbers, distances, total count.
    No manual capture step required.
    """
    fps_buffer = deque(maxlen=30)

    try:
        while True:
            start_time = time.time()

            frame, depth_map, err = grab_oak_frame(OAK_SESSION, wait_sec=0.1)

            if err or frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "OAK-D Camera Error", (150, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if err:
                    cv2.putText(placeholder, str(err)[:40], (100, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                ok, buffer = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                time.sleep(0.5)
                continue

            # Annotate with the latest auto-counted result from the background thread
            display_frame = frame.copy()
            det = latest_detection_result
            if det.get("sorted_bundles"):
                display_frame = draw_captured_bundles(
                    display_frame,
                    det["sorted_bundles"],
                    det["counted_bundle_ids"],
                    det["bundle_rebar_ranges"],
                )

                # Bottom info bar: total count + mode
                h, w = display_frame.shape[:2]
                mode_label = "NEAREST ONLY" if det["counting_mode"] == "nearest_only" else "ALL BUNDLES"
                summary = f"Total: {det['total_rebars']} rebars  |  Mode: {mode_label}"
                (sw, sh), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(display_frame, (0, h - 36), (w, h), (0, 0, 0), -1)
                cv2.putText(display_frame, summary, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # FPS overlay
            frame_time = time.time() - start_time
            fps_buffer.append(1.0 / max(frame_time, 0.001))
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                        (display_frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Timestamp
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


def draw_bundle_distances_only(frame, detection_result):
    """
    Draw ONLY bundle rectangles and distances on live feed - NO counting, NO rebar IDs.
    Bundles are labelled B1, B2, ... in order of detection (sorted by distance ascending).
    """
    if frame is None:
        return frame

    frame = frame.copy()
    bundles = detection_result.get("bundles", [])
    nearest_bundle = detection_result.get("nearest_bundle")

    if not bundles:
        return frame

    # Sort bundles by distance so labelling is consistent with capture output
    sorted_bundles = sorted(bundles, key=lambda b: b.get("distance_m") or float("inf"))

    # Colors for different bundles (index 0 = nearest = green)
    colors = [
        (0, 255, 0),    # Green  - nearest / B1
        (0, 255, 255),  # Yellow - B2
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (0, 165, 255),  # Orange
    ]

    for idx, bundle in enumerate(sorted_bundles):
        bounds = bundle.get("bounds")
        distance = bundle.get("distance_m")
        bundle_id = bundle.get("bundle_id")

        if not bounds or distance is None:
            continue

        x1, y1, x2, y2 = map(int, bounds)

        is_nearest = (nearest_bundle and bundle_id == nearest_bundle.get("bundle_id"))
        color = colors[idx % len(colors)]
        rect_thickness = 3 if is_nearest else 2

        # --- FIX 1: outer bounding box with B-label beside it ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, rect_thickness)

        bundle_label = f"B{idx + 1}"  # B1, B2, B3 … sequentially by distance
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.65
        label_thickness = 2
        (lw, lh), _ = cv2.getTextSize(bundle_label, font, label_scale, label_thickness)

        # Place label to the right of the top-right corner
        lx = x2 + 6
        ly = y1 + lh
        # Clamp so it stays inside the frame
        lx = min(lx, frame.shape[1] - lw - 4)

        cv2.rectangle(frame, (lx - 3, ly - lh - 3), (lx + lw + 3, ly + 3), (0, 0, 0), -1)
        cv2.putText(frame, bundle_label, (lx, ly), font, label_scale, color, label_thickness, cv2.LINE_AA)

        # Distance text above the rectangle
        distance_text = f"{distance:.2f}m"
        (tw, th), _ = cv2.getTextSize(distance_text, font, 0.6, 2)
        tx = x1 + (x2 - x1) // 2 - tw // 2
        ty = y1 - 10
        if ty < th + 10:
            ty = y2 + th + 10
        cv2.rectangle(frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
        cv2.putText(frame, distance_text, (tx, ty), font, 0.6, color, 2, cv2.LINE_AA)

        # NEAREST badge
        if is_nearest:
            cv2.putText(frame, "NEAREST", (x1, y1 - 5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Bottom info panel
    h, w = frame.shape[:2]
    panel_height = 80
    panel_y = h - panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    nearest_bundle_data = sorted_bundles[0] if sorted_bundles else None
    if nearest_bundle_data and nearest_bundle_data.get("distance_m"):
        cv2.putText(
            frame,
            f"NEAREST BUNDLE (B1): {nearest_bundle_data['distance_m']:.2f}m",
            (20, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Total Bundles: {len(sorted_bundles)}",
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


def draw_captured_bundles(frame, sorted_bundles, counted_bundle_ids, bundle_rebar_ranges=None):
    """
    Draw the post-capture annotated frame:
      - Outer bounding box per bundle labelled B1, B2, …
      - Each counted bundle's rebars are numbered independently starting from 1 by default.
      - If `bundle_rebar_ranges` is provided, numbers are global across counted bundles.
      - Uncounted bundles get a greyed-out box with their label only (no rebar numbers).

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate (will be copied).
    sorted_bundles : list[dict]
        Bundles sorted by distance ascending, each containing:
            bundle_id, distance_m, size, bounds, rebars (list of {box, …})
    counted_bundle_ids : set
        Which bundle_ids were selected for counting.
    bundle_rebar_ranges : dict or None
        Optional mapping of bundle_id -> (start, end) 1-based global rebar index.
    """
    if frame is None:
        return frame

    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    bundle_colors = [
        (0, 255, 0),    # B1 – green
        (0, 255, 255),  # B2 – yellow
        (255, 0, 255),  # B3 – magenta
        (255, 255, 0),  # B4 – cyan
        (0, 165, 255),  # B5 – orange
    ]
    uncounted_color = (100, 100, 100)  # grey for ignored bundles

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
        bundle_label = f"B{bundle_idx + 1}"  # always B1, B2, … regardless of counted status

        # --- FIX 1: outer bounding box + label beside it ---
        rect_thickness = 3 if is_counted else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, rect_thickness)

        (lw, lh), _ = cv2.getTextSize(bundle_label, font, 0.75, 2)
        lx = x2 + 6
        ly = y1 + lh
        lx = min(lx, frame.shape[1] - lw - 4)
        cv2.rectangle(frame, (lx - 3, ly - lh - 3), (lx + lw + 3, ly + 3), (0, 0, 0), -1)
        cv2.putText(frame, bundle_label, (lx, ly), font, 0.75, color, 2, cv2.LINE_AA)

        # Distance label above the box
        if distance is not None:
            dist_text = f"{distance:.2f}m"
            (tw, th), _ = cv2.getTextSize(dist_text, font, 0.55, 1)
            tx = x1 + (x2 - x1) // 2 - tw // 2
            ty = y1 - 10
            if ty < th + 10:
                ty = y2 + th + 10
            cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(frame, dist_text, (tx, ty), font, 0.55, color, 1, cv2.LINE_AA)

        if not is_counted:
            # Grey "IGNORED" watermark inside the box
            cv2.putText(
                frame, "IGNORED",
                (x1 + 6, y1 + 22),
                font, 0.55, uncounted_color, 1, cv2.LINE_AA,
            )
            continue

        # Rebar numbering (and mask) for counted bundles.
        bundle_range = bundle_rebar_ranges.get(bundle_id) if bundle_rebar_ranges else None
        for rebar_local_idx, rebar in enumerate(rebars):
            box = rebar.get("box")
            if box is None:
                continue

            rx1, ry1, rx2, ry2 = map(int, box)
            if bundle_range is not None:
                start_global = bundle_range[0]
                rebar_num = start_global + rebar_local_idx
            else:
                rebar_num = rebar_local_idx + 1  # 1-based, resets for every bundle

            # Fill count bundle mask for rebar (semi-transparent), only for counted bundles.
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, -1)
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

            # Rebar centre
            cx = (rx1 + rx2) // 2
            cy = (ry1 + ry2) // 2

            # Use a compact label size relative to rebar size
            rebar_w = abs(rx2 - rx1)
            rebar_h = abs(ry2 - ry1)
            label_size = max(12, int(min(rebar_w, rebar_h) * 0.35))
            radius = max(1, min(label_size, int(min(rebar_w, rebar_h) * 0.25)))

            # Outline circle for reference
            cv2.circle(frame, (cx, cy), radius, (0, 0, 0), 1)

            num_text = str(rebar_num)
            font_scale = max(0.12, min(0.35, float(label_size) / 32.0))
            thickness = 1
            (nw, nh), _ = cv2.getTextSize(num_text, font, font_scale, thickness)

            text_x = cx - nw // 2
            text_y = cy + nh // 2

            # Background rect only around text, small and non-overlapping where possible
            pad = max(1, int(label_size * 0.08))
            cv2.rectangle(
                frame,
                (text_x - pad, text_y - nh - pad),
                (text_x + nw + pad, text_y + pad),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                num_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return frame


@simple_router.post("/capture-and-count")
def capture_and_count(user_id: int = Form(...)):
    """
    Capture and count bundles with smart logic:
    - If distance difference between ANY bundles > 0.20m: Count ONLY nearest bundle
    - If distance difference between ALL consecutive bundles <= 0.20m: Count ALL bundles
    Returns annotated image where:
      - Every bundle has an outer bounding box labelled B1, B2, … beside it
      - Rebars inside counted bundles are numbered sequentially across ALL counted bundles
        (B1: 1..n, B2: n+1..m, …)
    Saves the detection to the database.
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

    # Get ALL bundles
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

    # If no depth-based distances, fall back to bundle-only mode (all bundles) instead of erroring.
    if not valid_bundles and bundles:
        sorted_bundles = bundles
        distance_mode = "no_depth"
    elif not valid_bundles:
        return {
            "total_rebars": 0,
            "counting_mode": "none",
            "bundles_counted": [],
            "all_bundles": [],
            "error": "No bundles with valid distances detected",
            "image": None
        }
    else:
        sorted_bundles = sorted(valid_bundles, key=lambda b: b["distance_m"])
        distance_mode = "depth"

    # Check distance differences between consecutive bundles (fallback for no depth is skipped)
    distance_diffs = []
    max_diff = 0

    if distance_mode == "depth" and len(sorted_bundles) > 1:
        for i in range(len(sorted_bundles) - 1):
            d1 = sorted_bundles[i].get("distance_m")
            d2 = sorted_bundles[i + 1].get("distance_m")
            if d1 is None or d2 is None:
                continue
            diff = abs(d1 - d2)
            distance_diffs.append({
                "bundle1_id": sorted_bundles[i]["bundle_id"],
                "bundle1_distance": d1,
                "bundle2_id": sorted_bundles[i + 1]["bundle_id"],
                "bundle2_distance": d2,
                "difference": round(diff, 2),
            })
            if diff > max_diff:
                max_diff = diff
    else:
        max_diff = 0
        distance_diffs = []

    # When there was no depth, choose fallback nearest-only mode instead of counting everything
    if distance_mode == "no_depth":
        max_diff = 0.0
        distance_diffs = []
        distance_mode = "no_depth_nearest"

    # Determine which bundles to count
    counted_bundle_ids = set()
    bundles_counted = []
    total_count = 0

    # Nearest-only conditions: explicit depth gap or no depth available
    if distance_mode in ["no_depth_nearest"] or max_diff > 0.20:
        # Count ONLY the nearest bundle (always B1 after sorting)
        if distance_mode == "no_depth_nearest":
            print(f"[Capture] No valid depth available (all unknown). Falling back to nearest-only")
        else:
            print(f"[Capture] Max distance difference = {max_diff:.2f}m > 0.20m")
        print(f"[Capture] Counting ONLY nearest bundle (B1)")

        nearest = sorted_bundles[0]
        counted_bundle_ids.add(nearest["bundle_id"])
        bundles_counted.append({
            "bundle_id": nearest["bundle_id"],
            "bundle_label": "B1",
            "distance_m": nearest["distance_m"],
            "rebar_count": nearest["size"],
            "display_text": f"B1 = {nearest['size']} bars",
            "counted": True
        })
        total_count = nearest["size"]
        counting_mode = "nearest_only"

        nearest_dist = f"{nearest['distance_m']:.2f}m" if nearest.get('distance_m') is not None else "n/a"
        print(f"  ✓ B1 (bundle {nearest['bundle_id']}): {nearest['size']} bars at {nearest_dist} (COUNTED)")
        for bi, bundle in enumerate(sorted_bundles[1:], start=2):
            bundle_dist = f"{bundle['distance_m']:.2f}m" if bundle.get('distance_m') is not None else "n/a"
            print(f"  ✗ B{bi} (bundle {bundle['bundle_id']}): {bundle['size']} bars at {bundle_dist} (IGNORED)")

    else:
        # Count ALL bundles
        print(f"[Capture] Max distance difference = {max_diff:.2f}m <= 0.20m")
        print(f"[Capture] Counting ALL bundles separately")

        for bi, bundle in enumerate(sorted_bundles, start=1):
            counted_bundle_ids.add(bundle["bundle_id"])
            bundles_counted.append({
                "bundle_id": bundle["bundle_id"],
                "bundle_label": f"B{bi}",
                "distance_m": bundle["distance_m"],
                "rebar_count": bundle["size"],
                "display_text": f"B{bi} = {bundle['size']} bars",
                "counted": True
            })
            total_count += bundle["size"]
            bundle_dist = f"{bundle['distance_m']:.2f}m" if bundle.get('distance_m') is not None else "n/a"
            print(f"  ✓ B{bi} (bundle {bundle['bundle_id']}): {bundle['size']} bars at {bundle_dist}")

        counting_mode = "all_bundles_separately"

    # Bundles detail for frontend (B-labels, independent counts per bundle)
    bundles_detail = []
    for bi, bundle in enumerate(sorted_bundles, start=1):
        bid = bundle["bundle_id"]
        bundles_detail.append({
            "bundle_id": bid,
            "bundle_label": f"B{bi}",
            "rebar_count": bundle["size"],
            "distance_m": bundle["distance_m"],
            "counted": bid in counted_bundle_ids,
        })

    # Draw annotated frame — use segmented base from the detector if available
    output_frame = frame
    if isinstance(annotated_rgb, np.ndarray):
        try:
            output_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            output_frame = frame

    annotated_result = draw_captured_bundles(
        output_frame, sorted_bundles, counted_bundle_ids
    )

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

    # Build display summary using B-labels
    if counting_mode == "nearest_only":
        display_summary = f"B1 = {bundles_counted[0]['rebar_count']} bars" if bundles_counted else "No bundles counted"
    else:
        display_summary = " | ".join([b["display_text"] for b in bundles_counted])

    response = {
        "det_id": det_id,
        "total_rebars": total_count,
        "counting_mode": counting_mode,
        "bundles_counted": bundles_counted,
        "bundles_detail": bundles_detail,
        "total_bundles": len([b for b in bundles_counted if b["counted"]]),
        "all_bundles": [
            {
                "bundle_id": b["bundle_id"],
                "bundle_label": f"B{bi}",
                "distance_m": b["distance_m"],
                "rebar_count": b["size"],
                "counted": b["bundle_id"] in counted_bundle_ids,
            }
            for bi, b in enumerate(sorted_bundles, start=1)
        ],
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
    print(f"  Bundles: {[b['bundle_label'] for b in bundles_counted]}")
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

    global detection_thread
    detection_thread = threading.Thread(target=continuous_detection, daemon=True)
    detection_thread.start()

    print("✅ Rebar-Counting backend started with:")
    print("   - Live feed: Auto-counts every frame — B1/B2 labels + sequential rebar numbers")
    print("   - No manual capture needed; counting runs continuously in background")
    print("   - Nearest-only mode when bundles are >0.20m apart, else all counted")
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
        "mode": "Auto-counting on live feed — B1/B2 labels + sequential rebar numbers, no capture step",
        "distance_tolerance_m": 0.20,
        "endpoints": {
            "live_stream": "/oak-stream",
            "capture_count": "/capture-and-count",
            "live_info": "/live-bundle-info",
            "snapshot": "/oak-snapshot"
        }
    }