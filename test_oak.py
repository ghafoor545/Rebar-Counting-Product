import time
import threading
import depthai as dai

def check_oak_camera(timeout_sec: float = 3.0) -> bool:
    """
    Check if an OAK camera is present and can produce frames.
    Returns True if working, False otherwise (never hangs).
    """
    # 1. Quick device presence check (non‑blocking)
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("[ERROR] No OAK device found.")
        return False
    print(f"[INFO] Found {len(devices)} OAK device(s).")

    # 2. Try to open a pipeline and capture one frame with timeout
    result = {"success": False, "frame": None}

    def attempt_open_and_grab():
        try:
            # Minimal pipeline
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setFps(30)

            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam.video.link(xout.input)

            device = dai.Device(pipeline)
            queue = device.getOutputQueue("rgb", maxSize=1, blocking=False)

            start = time.time()
            while time.time() - start < timeout_sec:
                if queue.has():
                    frame = queue.get().getCvFrame()
                    if frame is not None and frame.size > 0:
                        result["success"] = True
                        result["frame"] = frame
                        break
                time.sleep(0.02)
            device.close()
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")

    # Run the pipeline attempt in a separate thread
    thread = threading.Thread(target=attempt_open_and_grab)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_sec + 1.0)  # extra margin

    if result["success"]:
        print("[OK] Camera is working (frame received).")
        return True
    else:
        print("[ERROR] Timeout or failure – camera not responsive.")
        return False

if __name__ == "__main__":
    if check_oak_camera():
        print("Camera ready.")
    else:
        print("Camera not available.")