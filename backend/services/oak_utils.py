# backend/services/oak_utils.py

import time
import depthai as dai


def create_oak_pipeline():
    """
    Create DepthAI pipeline for OAK-D Pro RGB stream.
    Returns (pipeline, output_queue).
    """
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build()

    rgb_out = cam.requestOutput(
        size=(640, 480),
        type=dai.ImgFrame.Type.BGR888p  # BGR-packed
    )

    q_color = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    return pipeline, q_color


def ensure_oak_device(session):
    """
    Ensure an OAK pipeline + queue exist in session state.
    """
    if session.get("oak_device") is None or session.get("oak_queue") is None:
        pipeline, q_color = create_oak_pipeline()
        session["oak_device"] = pipeline
        session["oak_queue"] = q_color
    return session["oak_device"], session["oak_queue"]


def get_oak_frame(pipeline, queue):
    """
    Non-blocking get of a single OAK frame as BGR numpy array.
    """
    if hasattr(pipeline, "isRunning") and not pipeline.isRunning():
        return None

    try:
        if queue.has():
            in_frame = queue.get()
            return in_frame.getCvFrame()  # BGR numpy array
    except Exception:
        return None

    return None


def grab_oak_frame(session, wait_sec: float = 2.0):
    """
    Try to grab one frame within wait_sec seconds.
    Used by 'Capture & Count' for OAK-D Pro.
    """
    try:
        pipeline, queue = ensure_oak_device(session)
    except Exception as e:
        return None, f"Failed to initialize OAK-D Pro: {e}"

    t0 = time.time()
    frame = None
    while time.time() - t0 < wait_sec:
        frame = get_oak_frame(pipeline, queue)
        if frame is not None:
            break
        time.sleep(0.03)

    if frame is None:
        return None, "Failed to grab frame from OAK-D Pro."

    return frame, None


def close_oak_device(session):
    """
    Stop and clear the OAK pipeline from session state.
    """
    try:
        if session.get("oak_device") is not None:
            if hasattr(session["oak_device"], "stop"):
                session["oak_device"].stop()
    except Exception:
        pass

    session["oak_device"] = None
    session["oak_queue"] = None