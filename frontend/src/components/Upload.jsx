// src/components/Upload.jsx
import React, { useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function Upload({ user, onSuccess }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    if (!file) {
      setResult({ error: "Please select an image file." });
      return;
    }

    if (!user || !user.id) {
      setResult({ error: "User not found. Please sign in again." });
      return;
    }

    setBusy(true);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("user_id", String(user.id));
      fd.append("stream_url", "Upload");

      const res = await fetch(`${API_BASE_URL}/detections/upload`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok || data.error) {
        throw new Error(data.error || data.detail || "Detection error");
      }

      setResult(data);

      if (onSuccess) onSuccess();
    } catch (err) {
      setResult({ error: err.message || "Detection error" });
    } finally {
      setBusy(false);
    }
  };

  const computeUploadSummary = (result) => {
    const bundleInfo = result?.bundle_info || {};
    const bundles = Array.isArray(bundleInfo.bundles) ? bundleInfo.bundles : [];

    const validBundles = bundles
      .filter((b) => b && b.bundle_id != null)
      .sort((a, b) => {
        const da = typeof a.distance_m === "number" ? a.distance_m : Number.MAX_VALUE;
        const db = typeof b.distance_m === "number" ? b.distance_m : Number.MAX_VALUE;
        return da - db;
      });

    const distanceAware = validBundles.filter((b) => typeof b.distance_m === "number");

    let maxDiff = 0;
    for (let i = 0; i < distanceAware.length - 1; i++) {
      const diff = Math.abs(distanceAware[i].distance_m - distanceAware[i + 1].distance_m);
      if (diff > maxDiff) maxDiff = diff;
    }

    let mode;
    if (validBundles.length <= 1) {
      mode = "nearest_only";
    } else if (distanceAware.length >= 2) {
      mode = maxDiff > 0.2 ? "nearest_only" : "all_bundles_separately";
    } else {
      mode = "all_bundles_separately";
    }

    const countedBundles =
      mode === "nearest_only" && validBundles.length > 0
        ? [validBundles[0]]
        : validBundles;

    const bundleText = countedBundles.length > 0
      ? countedBundles
          .map((b, idx) => {
            const count = b.size ?? (Array.isArray(b.rebars) ? b.rebars.length : 0);
            return `B${idx + 1} = ${count} bars`;
          })
          .join(" | ")
      : "";

    return {
      total_rebars: result?.count ?? 0,
      total_bundles: validBundles.length,
      max_distance_difference: Number(maxDiff.toFixed(2)),
      counting_mode: mode,
      countedBundles,
      bundleText,
    };
  };

  const renderMessage = () => {
    if (!result) return null;
    if (result.error) {
      return <div className="alert error">{result.error}</div>;
    }

    const summary = computeUploadSummary(result);
    const modeMessage =
      summary.counting_mode === "nearest_only"
        ? "🔍 Mode: Nearest Only"
        : "📊 Mode: All Bundles";
    const distanceMessage =
      summary.total_bundles <= 1
        ? "Only one bundle detected."
        : summary.max_distance_difference > 0.2
        ? `Distance difference (${summary.max_distance_difference.toFixed(2)}m) > 0.20m → Counted only nearest bundle`
        : `All distance differences ≤ 0.20m → Counted all bundles separately`;

    const bundleListMessage = summary.bundleText
      ? `Counted Bundles: ${summary.bundleText}`
      : "No bundles counted.";

    return (
      <div className="alert success">
        <div className="result-header">
          <div className="result-count">🎯 Total: {summary.total_rebars} rebars</div>
          <div className="result-bundles">📦 Bundles: {summary.total_bundles}</div>
          <div className="result-mode">{modeMessage}</div>
        </div>

        <div className="mode-reason">{distanceMessage}</div>
        <div className="bundle-summary"><strong>📋 {bundleListMessage}</strong></div>

        {summary.countedBundles && summary.countedBundles.length > 0 && (
          <div className="bundle-details">
            <h5>📊 BUNDLE DETAILS</h5>
            <ul>
              {summary.countedBundles.map((b, idx) => {
                const count = b.size ?? (Array.isArray(b.rebars) ? b.rebars.length : 0);
                return (
                  <li key={`${b.bundle_id || idx}-${idx}`}>
                    B{idx + 1} (bundle {b.bundle_id ?? "?"}): {count} bars
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <div className="grid-row">
        <div className="grid-col">
          <label className="form-label">
            Upload image (JPG/PNG)
            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={(e) => setFile(e.target.files[0] || null)}
            />
          </label>
        </div>
        <div className="grid-col small">
          <button
            className="btn primary"
            type="button"
            onClick={handleSubmit}
            disabled={busy}
          >
            {busy ? "Detecting..." : "Detect"}
          </button>
        </div>
      </div>

      {renderMessage()}

      {result && result.image && (
        <div className="captured-image-wrap">
          <img
            src={result.image}
            alt="Annotated result"
            className="captured-image-full"
          />
        </div>
      )}
    </div>
  );
}

export default Upload;