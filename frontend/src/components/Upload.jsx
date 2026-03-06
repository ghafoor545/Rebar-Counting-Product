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

  const renderMessage = () => {
    if (!result) return null;
    if (result.error) {
      return <div className="alert error">{result.error}</div>;
    }

    const bi = result.bundle_info || {};

    // ✅ Support both keys:
    // - total_bundles / total_rebars_in_bundles / total_isolated (live bundle_info)
    // - rebars_in_bundles / isolated (DB-minified JSON)
    const bundles = bi.total_bundles ?? 0;
    const inBundles =
      bi.rebars_in_bundles ??
      bi.total_rebars_in_bundles ??
      0;
    const isolated =
      bi.total_isolated ??
      bi.isolated ??
      0;

    if (bundles > 0) {
      return (
        <div className="alert success">
          Count: {inBundles} | Bundles: {bundles} | Isolated: {isolated}
        </div>
      );
    }

    return (
      <div className="alert success">
        Count: 0 | Isolated: {isolated}
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