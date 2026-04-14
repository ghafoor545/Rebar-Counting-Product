# backend/utils/helpers.py


from typing import Any, Dict, Optional, Tuple

# Simple in-memory session-like storage to emulate flash behavior
_session_state: Dict[str, Any] = {}


def do_rerun():
    """
    Original logic attempted to trigger a Streamlit rerun.
    In a FastAPI backend context, there is no rerun mechanism,
    so this is implemented as a no-op placeholder.
    """
    return None


def scroll_top():
    """
    Original logic injected JS to scroll the page to top.
    In an API backend context, there is no client DOM to manipulate,
    so this is implemented as a no-op placeholder.
    """
    return None


def show_image_full_width(img, caption: Optional[str] = None):
    """
    Original logic displayed an image via Streamlit at full width.

    Here, we simply return the image (and caption) so callers can decide
    how to handle or serialize it (e.g. encode to base64 and return in JSON).
    """
    return {"image": img, "caption": caption}


def flash(kind: str, text: str):
    """
    Store a one-time flash message in a simple in-memory session state.

    Behavior mirrors the original idea (store under '_flash' key).
    """
    _session_state["_flash"] = (kind, str(text))


def show_flash() -> Optional[Dict[str, str]]:
    """
    Retrieve and clear the stored flash message, if any.

    Returns:
        - None if no flash is stored.
        - dict with keys: 'kind' and 'text' if a flash was present.
    """
    f: Optional[Tuple[str, str]] = _session_state.pop("_flash", None)
    if not f:
        return None

    kind, text = f
    return {"kind": kind, "text": str(text)}