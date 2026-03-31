# backend/services/oak_utils.py

import time
import numpy as np
import cv2

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False


def create_oak_pipeline():
    """Create OAK-D Pro pipeline with RGB + Depth."""
    if not DEPTHAI_AVAILABLE:
        raise RuntimeError("depthai not available")
    
    pipeline = dai.Pipeline()

    # RGB Camera
    cam = pipeline.create(dai.node.Camera).build()
    rgb_out = cam.requestOutput(
        size=(640, 480),
        type=dai.ImgFrame.Type.BGR888p
    )
    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    # Stereo Depth
    q_depth = None
    try:
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        depth_out = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        q_depth = depth_out
        
        print("[OAK] ✓ Depth enabled")
    except Exception as e:
        print(f"[OAK] Depth disabled: {e}")

    pipeline.start()
    print("[OAK] ✓ Pipeline started")
    
    return pipeline, q_rgb, q_depth


def ensure_oak_device(session):
    """Ensure OAK device is initialized."""
    if session.get("oak_device") is None or session.get("oak_queue_rgb") is None:
        pipeline, q_rgb, q_depth = create_oak_pipeline()
        session["oak_device"] = pipeline
        session["oak_queue_rgb"] = q_rgb
        session["oak_queue_depth"] = q_depth
    
    return (
        session["oak_device"],
        session["oak_queue_rgb"],
        session.get("oak_queue_depth")
    )


def get_oak_frame(pipeline, queue_rgb, queue_depth):
    """Non-blocking get of RGB + Depth frames."""
    rgb_frame = None
    depth_map = None
    
    if hasattr(pipeline, "isRunning") and not pipeline.isRunning():
        return None, None

    try:
        if queue_rgb and queue_rgb.has():
            rgb_frame = queue_rgb.get().getCvFrame()
        
        if queue_depth:
            try:
                if queue_depth.has():
                    depth_map = queue_depth.get().getFrame()
                    if depth_map is not None and rgb_frame is not None:
                        if depth_map.shape != rgb_frame.shape[:2]:
                            depth_map = cv2.resize(
                                depth_map,
                                (rgb_frame.shape[1], rgb_frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
            except Exception:
                pass
    except Exception:
        return None, None

    return rgb_frame, depth_map


def grab_oak_frame(session, wait_sec: float = 2.0):
    """Grab RGB + Depth with timeout. Returns (rgb, depth, error)."""
    if not DEPTHAI_AVAILABLE:
        return None, None, "depthai not installed"
    
    try:
        pipeline, queue_rgb, queue_depth = ensure_oak_device(session)
    except Exception as e:
        return None, None, f"Init failed: {e}"

    t0 = time.time()
    rgb_frame = None
    depth_map = None
    
    while time.time() - t0 < wait_sec:
        rgb_frame, depth_map = get_oak_frame(pipeline, queue_rgb, queue_depth)
        if rgb_frame is not None:
            break
        time.sleep(0.03)

    if rgb_frame is None:
        return None, None, "Timeout: no frame"

    return rgb_frame, depth_map, None


def close_oak_device(session):
    """Close OAK device."""
    try:
        if session.get("oak_device"):
            if hasattr(session["oak_device"], "stop"):
                session["oak_device"].stop()
            print("[OAK] ✓ Device closed")
    except Exception:
        pass
    
    session["oak_device"] = None
    session["oak_queue_rgb"] = None
    session["oak_queue_depth"] = None


def get_depth_at_point(depth_map: np.ndarray, x: int, y: int, sample_radius: int = 5) -> float:
    """Get median depth at a point (in mm)."""
    if depth_map is None or depth_map.size == 0:
        return float('inf')
    
    h, w = depth_map.shape[:2]
    y1, y2 = max(0, y - sample_radius), min(h, y + sample_radius + 1)
    x1, x2 = max(0, x - sample_radius), min(w, x + sample_radius + 1)
    
    if y1 >= y2 or x1 >= x2:
        return float('inf')
    
    region = depth_map[y1:y2, x1:x2]
    valid = region[(region > 100) & (region < 10000)]
    
    return float(np.median(valid)) if len(valid) > 0 else float('inf')


def overlay_depth_on_live_feed(frame_bgr, depth_map, detector, model):
    """Overlay green boxes + depth on live feed."""
    if frame_bgr is None:
        return frame_bgr
    
    img = frame_bgr.copy()
    
    if depth_map is None:
        cv2.putText(img, "No Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    
    try:
        sess = model["sess"]
        in_name = model["in_name"]
        in_hw = model["in_hw"]
        
        blob, r, dwdh = detector.preprocess_for_onnx(frame_bgr, in_hw)
        outs = sess.run(None, {in_name: blob})
        kind, data = detector.parse_onnx_outputs(outs)
        
        boxes = []
        
        if kind == "seg":
            pred_raw, _ = data
            pred = pred_raw[0]
            if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
                pred = pred.T
            
            pred = pred.astype(np.float32)
            b = pred[:, 0:4].copy()
            scores = detector.maybe_sigmoid(pred[:, 4:5])
            if scores.ndim > 1:
                scores = scores[:, 0]
            
            in_h, in_w = in_hw
            if np.nanmax(b) <= 1.5:
                b[:, [0, 2]] *= in_w
                b[:, [1, 3]] *= in_h
            
            b = detector.xywh2xyxy(b)
            keep = scores >= 0.5
            b, scores = b[keep], scores[keep]
            
            b = detector.scale_boxes(b, r, dwdh, frame_bgr.shape)
            
            keep_nms = detector.nms(b, scores, iou_thres=0.3)
            if keep_nms:
                boxes = b[keep_nms]
        
        # Draw boxes + depth
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            depth_mm = get_depth_at_point(depth_map, cx, cy)
            depth_m = depth_mm / 1000.0
            
            # Green box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Depth label
            if depth_mm < 9999:
                text = f"{depth_m:.2f}m"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
                
                ly = y1 - 6 if y1 > 25 else y2 + 16
                cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), (0, 100, 255), -1)
                cv2.putText(img, text, (x1 + 3, ly - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(img, f"Detected: {len(boxes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception:
        pass
    
    return img
