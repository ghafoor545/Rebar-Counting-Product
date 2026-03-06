# backend/oled_display.py
import threading
from datetime import datetime

try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import sh1106
except Exception:
    i2c = None
    canvas = None
    sh1106 = None


class OledStatus:
    """
    Wrapper around SH1106 OLED used by the Rebar app.

    Design goals:
    - Only update the display when app state changes (no constant ticking).
    - Always clear the full visible area.
    - Show time on the "Rebar Count" screen.
    """

    def __init__(self, port: int = 1, address: int = 0x3C):
        self._lock = threading.Lock()
        self._last_payload = None

        if i2c is None or canvas is None or sh1106 is None:
            self.display = None
            return

        serial = i2c(port=port, address=address)
        self.display = sh1106(serial)
        self.display.clear()

    def _draw_lines(self, lines):
        """
        Draw up to 4 lines of text.

        - Clears entire visible screen (0..width, 0..height).
        - Skips drawing if text is unchanged from last call.
        """
        if self.display is None:
            return

        payload = tuple(lines)
        if payload == self._last_payload:
            return
        self._last_payload = payload

        w = self.display.width
        h = self.display.height

        with self._lock:
            with canvas(self.display) as draw:
                # Clear full visible area
                draw.rectangle((0, 0, w, h), outline=0, fill=0)

                # Draw lines with vertical spacing
                y = 4
                for line in lines[:4]:
                    draw.text((2, y), line[:20], fill="white")
                    y += 18

    # ---- Screens used by the app ----

    def show_ready(self):
        # Ready screen: no time, static
        self._draw_lines(
            [
                "        NUTECH",
                "      Rebar App",
                "        Ready",
            ]
        )

    def show_processing(self):
        # While detection is running
        self._draw_lines(
            [
                "Processing",
                "Counting...",
            ]
        )

    def show_count(self, count: int):
        # Show time at moment of last count (no ticking)
        t = datetime.now().strftime("%H:%M:%S")
        self._draw_lines(
            [
                t,
                f"Count: {count}",
            ]
        )

    def show_message(self, line1: str, line2: str = "", line3: str = ""):
        lines = [line1]
        if line2:
            lines.append(line2)
        if line3:
            lines.append(line3)
        self._draw_lines(lines)

    def clear(self):
        if self.display is not None:
            with self._lock:
                self.display.clear()
            self._last_payload = None


# --- Singleton helpers for easy use from application code ---

_instance = None
_init_error = None


def _get_oled():
    global _instance, _init_error
    if _instance is None and _init_error is None:
        try:
            _instance = OledStatus()
        except Exception as e:
            _init_error = e
    return _instance


def oled_show_ready():
    dev = _get_oled()
    if dev:
        dev.show_ready()


def oled_show_processing():
    dev = _get_oled()
    if dev:
        dev.show_processing()


def oled_show_count(count: int):
    dev = _get_oled()
    if dev:
        dev.show_count(count)


def oled_show_message(line1: str, line2: str = "", line3: str = ""):
    dev = _get_oled()
    if dev:
        dev.show_message(line1, line2, line3)


def oled_clear():
    dev = _get_oled()
    if dev:
        dev.clear()