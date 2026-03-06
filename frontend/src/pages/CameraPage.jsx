// src/pages/CameraPage.jsx
import React, { useEffect, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function CameraPage({ user }) {
  const [streaming, setStreaming] = useState(false);
  const [liveWidth, setLiveWidth] = useState(640);
  const [captureResult, setCaptureResult] = useState(null);
  const [busyCapture, setBusyCapture] = useState(false);
  const [galleryItems, setGalleryItems] = useState([]);

  // Fetch recent detections for gallery
  const fetchRecent = async () => {
    if (!user || !user.id) return;
    try {
      const url = new URL(`${API_BASE_URL}/detections/recent`);
      url.searchParams.set("user_id", user.id);
      url.searchParams.set("limit", "10");
      // NOTE: backend doesn't filter by "source", so this param is ignored currently
      url.searchParams.set("source", "OAK-D Pro");
      const res = await fetch(url.toString());
      if (!res.ok) return;
      const data = await res.json().catch(() => ({}));
      setGalleryItems(data.items || []);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    fetchRecent();
  }, [user]);

  const handleStartLive = () => setStreaming(true);
  const handleStopLive = () => setStreaming(false);

  const handleCaptureAndCount = async () => {
    if (!user || !user.id) return;
    setBusyCapture(true);
    try {
      // Grab one frame from OAK backend stream
      const snapRes = await fetch(`${API_BASE_URL}/oak-snapshot`);
      if (!snapRes.ok) throw new Error("Failed to grab frame from OAK-D.");

      const blob = await snapRes.blob();
      const file = new File([blob], "oak_capture.jpg", { type: "image/jpeg" });

      const fd = new FormData();
      fd.append("user_id", String(user.id));
      fd.append("file", file);
      fd.append("stream_url", "OAK-D Pro");

      const res = await fetch(`${API_BASE_URL}/detections/upload`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok || data.error) {
        throw new Error(data.error || data.detail || "Capture failed.");
      }

      setCaptureResult(data);
      // refresh gallery after successful capture
      fetchRecent();
    } catch (e) {
      setCaptureResult({ error: e.message || "Capture failed." });
    } finally {
      setBusyCapture(false);
    }
  };

  // Helper to parse bundle_info from live result (captureResult)
  const parseBundleInfoLive = (biRaw) => {
    const bi = biRaw || {};
    const bundles = bi.total_bundles ?? 0;
    const inBundles =
      bi.rebars_in_bundles ??
      bi.total_rebars_in_bundles ??
      0;
    const isolated =
      bi.isolated ??
      bi.total_isolated ??
      0;
    return {
      bundles: Number(bundles) || 0,
      inBundles: Number(inBundles) || 0,
      isolated: Number(isolated) || 0,
    };
  };

  // Message under the "Capture & Count" card
  const renderMessage = () => {
    if (!captureResult) return null;
    if (captureResult.error) {
      return <div className="alert error">{captureResult.error}</div>;
    }

    const { bundles, inBundles, isolated } = parseBundleInfoLive(
      captureResult.bundle_info
    );

    // ✅ Use captureResult.count (what detector saved) as primary Count
    const countFromResult =
      typeof captureResult.count === "number" ? captureResult.count : 0;

    const displayCount =
      countFromResult > 0 ? countFromResult : inBundles + isolated;

    return (
      <div className="alert success">
        Count: {displayCount} | Bundles: {bundles}
        {isolated > 0 ? ` | Isolated: ${isolated}` : ""}
      </div>
    );
  };

  // Helper to parse bundle_info from DB for gallery
  const parseBundleInfoDb = (raw) => {
    let bi = raw;
    if (typeof bi === "string") {
      try {
        bi = JSON.parse(bi);
      } catch {
        bi = {};
      }
    }
    bi = bi || {};
    const bundles = bi.total_bundles ?? 0;
    const inBundles =
      bi.rebars_in_bundles ??
      bi.total_rebars_in_bundles ??
      0;
    const isolated =
      bi.isolated ??
      bi.total_isolated ??
      0;
    return {
      bundles: Number(bundles) || 0,
      inBundles: Number(inBundles) || 0,
      isolated: Number(isolated) || 0,
    };
  };

  return (
    <div className="page-wrap">
      {/* Live control card */}
      <div className="dashboard-wrap">
        <div className="card">
          <h3
            className="gradient-heading"
            style={{ margin: "0 0 8px 0" }}
          >
            Live Camera
          </h3>
          <div className="grid-row">
            <div className="grid-col">
              <label className="form-label">
                Live Width
                <input
                  type="number"
                  value={liveWidth}
                  onChange={(e) =>
                    setLiveWidth(Number(e.target.value) || 640)
                  }
                />
              </label>
            </div>
            <div className="grid-col small">
              <button className="btn primary" onClick={handleStartLive}>
                Start Live
              </button>
            </div>
            <div className="grid-col small">
              <button className="btn secondary" onClick={handleStopLive}>
                Stop Live
              </button>
            </div>
          </div>
          <div className="status-row">
            <span
              className={`status-chip ${streaming ? "on" : "off"}`}
            >
              Live: {streaming ? "ON" : "OFF"}
            </span>
          </div>
        </div>
      </div>

      {/* Live stream + capture card */}
      <div className="dashboard-wrap">
        <div className="card live-card">
          <div className="live-wrap">
            <div
              className="live-frame"
              style={{ width: `${liveWidth}px`, maxWidth: "96vw" }}
            >
              <div className="live-inner">
                {streaming ? (
                  <img
                    src={`${API_BASE_URL}/oak-stream`}
                    className="live-img"
                    alt="stream"
                  />
                ) : (
                  <div className="live-placeholder">
                    Live stream OFF
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="capture-btn-row">
            <button
              className="btn primary"
              onClick={handleCaptureAndCount}
              disabled={busyCapture}
            >
              {busyCapture ? "Processing..." : "Capture & Count"}
            </button>
          </div>
        </div>
      </div>

      {/* Last capture result card */}
      <div className="dashboard-wrap">
        <div className="card">
          {renderMessage()}
          {captureResult?.image && (
            <div className="captured-image-wrap">
              <img
                src={captureResult.image}
                className="captured-image-full"
                alt="res"
              />
            </div>
          )}
        </div>
      </div>

      {/* Camera gallery */}
      <div className="dashboard-wrap">
        <div className="card gallery-card">
          <h4
            className="gradient-heading"
            style={{ margin: "0 0 12px 0", fontSize: "1.1rem" }}
          >
            Recent Camera Captures
          </h4>
          {galleryItems.length > 0 ? (
            <div className="shot-grid">
              {galleryItems.map((item) => {
                const { bundles, inBundles, isolated } = parseBundleInfoDb(
                  item.bundle_info
                );

                // ✅ Use DB count as primary Count
                const countFromDb =
                  typeof item.count === "number" ? item.count : 0;

                // Fallback: if DB count was ever 0, use bundle_info totals
                const fromBundle = inBundles + isolated;
                const displayCount =
                  countFromDb > 0 ? countFromDb : fromBundle;

                const text = `Count: ${displayCount} | B:${bundles}${
                  isolated > 0 ? ` I:${isolated}` : ""
                }`;

                return (
                  <div className="shot-card" key={item.id}>
                    <span className="shot-badge">{text}</span>
                    {item.image && (
                      <img src={item.image} alt="detection" />
                    )}
                    <div className="shot-footer">{text}</div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "#d7e2ff" }}>
              No camera captures yet.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CameraPage;