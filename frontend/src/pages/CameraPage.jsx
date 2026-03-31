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

  const userId = user?.id || 1;

  // Fetch recent detections for gallery
  const fetchRecent = async () => {
    try {
      const url = new URL(`${API_BASE_URL}/detections/recent`);
      url.searchParams.set("user_id", userId);
      url.searchParams.set("limit", "10");
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
    setBusyCapture(true);
    try {
      const formData = new FormData();
      formData.append("user_id", String(userId));

      const res = await fetch(`${API_BASE_URL}/capture-and-count`, {
        method: "POST",
        body: formData,
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

  // Parse bundle info from capture result (UPDATED)
  const parseBundleInfoFromCapture = (result) => {
    if (!result) return null;
    
    // If we have bundles_detail from backend, use that
    if (result.bundles_detail && result.bundles_detail.length > 0) {
      return {
        bundles_counted: result.bundles_detail,
        total_bundles: result.total_bundles || result.bundles_detail.length,
        counting_mode: result.counting_mode,
        total_rebars: result.total_rebars
      };
    }
    
    // Fallback to parsing bundles_counted
    if (result.bundles_counted && result.bundles_counted.length > 0) {
      return {
        bundles_counted: result.bundles_counted,
        total_bundles: result.bundles_counted.length,
        counting_mode: result.counting_mode,
        total_rebars: result.total_rebars
      };
    }
    
    return null;
  };

  // Render detailed bundle information
  const renderDetailedBundles = () => {
    const bundleInfo = parseBundleInfoFromCapture(captureResult);
    if (!bundleInfo || !bundleInfo.bundles_counted || bundleInfo.bundles_counted.length === 0) {
      return null;
    }
    
    const bundleColors = ['#00ff00', '#ffa500', '#ff00ff', '#00ffff', '#ffff00'];
    
    return (
      <div className="bundle-breakdown">
        <h5>📊 BUNDLE DETAILS</h5>
        <div className="bundle-list">
          {bundleInfo.bundles_counted.map((bundle, idx) => {
            const counted = !!bundle.counted;
            const statusText = counted ? "✓ COUNTED" : "⛔ IGNORED";
            const statusClass = counted ? "bundle-status counted" : "bundle-status ignored";
            const label = bundle.bundle_label || `B${idx + 1}`;

            return (
              <div key={`${bundle.bundle_id}-${idx}`} className="bundle-item">
                <div className="bundle-dot" style={{ backgroundColor: bundleColors[idx % bundleColors.length] }}></div>
                <div className="bundle-name">{label}</div>
                <div className="bundle-size">{bundle.rebar_count} rebars</div>
                <div className="bundle-depth">
              📏 {bundle.distance_m != null && !Number.isNaN(bundle.distance_m)
                ? `${bundle.distance_m.toFixed(2)}m`
                : "unknown"}
            </div>
                <div className={statusClass}>{statusText}</div>
              </div>
            );
          })}
        </div>
        
        <div className="total-rebars">
          <strong>🎯 TOTAL REBARS COUNTED: {bundleInfo.total_rebars}</strong>
        </div>
      </div>
    );
  };

  // Parse message for capture result (UPDATED)
  const renderMessage = () => {
    if (!captureResult) return null;
    if (captureResult.error) {
      return <div className="alert error">{captureResult.error}</div>;
    }

    const bundleInfo = parseBundleInfoFromCapture(captureResult);
    const totalBundles = bundleInfo?.total_bundles || 0;
    const countingMode = captureResult.counting_mode || (totalBundles > 0 ? (totalBundles === 1 ? "nearest_only" : "all_bundles_separately") : "none");
    const maxDiff = captureResult.max_distance_difference || 0;
    
    // Create display text for bundles (only counted ones)
    let bundleDisplayText = "";
    if (bundleInfo && bundleInfo.bundles_counted) {
      const countedBundles = bundleInfo.bundles_counted.filter((b) => b.counted);
      bundleDisplayText = countedBundles.map((b) => `Bundle ${b.bundle_id} = ${b.rebar_count} bars`).join(" | ");
    }
    
    return (
      <div className="alert success">
        <div className="result-header">
          <div className="result-count">
            🎯 Total: {captureResult.total_rebars || 0} rebars
          </div>
          <div className="result-bundles">
            📦 Bundles: {totalBundles}
          </div>
          <div className="result-mode">
            {countingMode === "nearest_only" ? "🔍 Mode: Nearest Only" : "📊 Mode: All Bundles"}
          </div>
        </div>
        <div className="mode-reason">
          {maxDiff > 0.20 
            ? `Distance difference (${maxDiff.toFixed(2)}m) > 0.20m → Counted only nearest bundle`
            : `All distance differences ≤ 0.20m → Counted all bundles separately`}
        </div>
        
        {/* Show individual bundle details */}
        {bundleDisplayText && (
          <div className="bundle-summary">
            <strong>📋 Counted Bundles:</strong> {bundleDisplayText}
          </div>
        )}
        
        {renderDetailedBundles()}
      </div>
    );
  };

  // Parse bundle_info from DB for gallery
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
    const inBundles = bi.rebars_in_bundles ?? bi.total_rebars_in_bundles ?? 0;
    const isolated = bi.isolated ?? bi.total_isolated ?? 0;
    const nearestBundle = bi.nearest_bundle;
    const bundleDetails = bi.bundles || [];
    
    return {
      bundles: Number(bundles) || 0,
      inBundles: Number(inBundles) || 0,
      isolated: Number(isolated) || 0,
      nearestDistance: nearestBundle?.distance_m,
      bundleDetails: bundleDetails
    };
  };

  return (
    <div className="page-wrap">
      {/* Live control card */}
      <div className="dashboard-wrap">
        <div className="card">
          <h3 className="gradient-heading" style={{ margin: "0 0 8px 0" }}>
            Live Camera
          </h3>
          <div className="grid-row">
            <div className="grid-col">
              <label className="form-label">
                Live Width
                <input
                  type="number"
                  value={liveWidth}
                  onChange={(e) => setLiveWidth(Number(e.target.value) || 640)}
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
            <span className={`status-chip ${streaming ? "on" : "off"}`}>
              Live: {streaming ? "ON" : "OFF"}
            </span>
          </div>
        </div>
      </div>

      {/* Live stream + capture card */}
      <div className="dashboard-wrap">
        <div className="card live-card">
          <div className="live-wrap">
            <div className="live-frame" style={{ width: `${liveWidth}px`, maxWidth: "96vw" }}>
              <div className="live-inner">
                {streaming ? (
                  <img src={`${API_BASE_URL}/oak-stream`} className="live-img" alt="stream" />
                ) : (
                  <div className="live-placeholder">Live stream OFF</div>
                )}
              </div>
            </div>
          </div>
          <div className="capture-btn-row">
            <button className="btn primary" onClick={handleCaptureAndCount} disabled={busyCapture}>
              {busyCapture ? "Processing..." : "Capture & Count"}
            </button>
          </div>
        </div>
      </div>

      {/* Last capture result card */}
      <div className="dashboard-wrap">
        <div className="card result-card">
          <h4 className="gradient-heading" style={{ margin: "0 0 12px 0" }}>
            📸 Last Capture Result
          </h4>
          {renderMessage()}
          {captureResult?.image && (
            <div className="captured-image-wrap">
              <img src={captureResult.image} className="captured-image-full" alt="result" />
            </div>
          )}
        </div>
      </div>

      {/* Camera gallery */}
      <div className="dashboard-wrap">
        <div className="card gallery-card">
          <h4 className="gradient-heading" style={{ margin: "0 0 12px 0", fontSize: "1.1rem" }}>
            📷 Recent Camera Captures
          </h4>
          {galleryItems.length > 0 ? (
            <div className="shot-grid">
              {galleryItems.map((item) => {
                const { bundles, inBundles, isolated, nearestDistance, bundleDetails } = parseBundleInfoDb(item.bundle_info);
                const countFromDb = typeof item.count === "number" ? item.count : 0;
                const fromBundle = inBundles + isolated;
                const displayCount = countFromDb > 0 ? countFromDb : fromBundle;
                
                // Create display text with individual bundle info
                let displayText = `Count: ${displayCount} | B:${bundles}`;
                if (nearestDistance) {
                  displayText += ` | Nearest: ${nearestDistance.toFixed(2)}m`;
                }
                if (isolated > 0) {
                  displayText += ` | I:${isolated}`;
                }
                
                // Show individual bundle details
                const bundleDetailsText = bundleDetails.length > 0 
                  ? bundleDetails.map(b => `B${b.bundle_id}:${b.size}`).join(', ')
                  : '';

                return (
                  <div className="shot-card" key={item.id}>
                    <span className="shot-badge">{displayText}</span>
                    {bundleDetailsText && (
                      <span className="shot-mode-badge">{bundleDetailsText}</span>
                    )}
                    {item.image && <img src={item.image} alt="detection" />}
                    <div className="shot-footer">
                      {displayText}
                      {bundleDetailsText && <div className="shot-detail">{bundleDetailsText}</div>}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "#d7e2ff" }}>No camera captures yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CameraPage;