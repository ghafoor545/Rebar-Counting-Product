# backend/utils/utils.py

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fmt_local_time(ts: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        if ts.endswith("Z"):
            dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt_utc = datetime.fromisoformat(ts)

        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        local_tz = datetime.now().astimezone().tzinfo
        return dt_utc.astimezone(local_tz).strftime(fmt)
    except Exception:
        return ts.replace("T", " ")[:19]