// src/pages/UploadPage.jsx
import React, { useEffect, useState } from "react";
import Upload from "../components/Upload";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function UploadPage({ user }) {
  const [galleryItems, setGalleryItems] = useState([]);
  const userId = user?.id || 1;

  // Fetch recent gallery (limit 10)
  const fetchRecent = async () => {
    try {
      const url = new URL(`${API_BASE_URL}/detections/recent`);
      url.searchParams.set("user_id", userId);
      url.searchParams.set("limit", "10");
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

  // Callback to refresh gallery after a successful upload
  const handleUploadSuccess = () => {
    fetchRecent();
  };

  // Parse bundle_info from DB/string
  const parseBundleInfo = (raw) => {
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

  const computeUploadSummary = (result) => {
    const bundleInfo = result?.bundle_info || {};
    const bundles = Array.isArray(bundleInfo.bundles) ? bundleInfo.bundles : [];

    const validBundles = bundles
      .filter((b) => b && typeof b.distance_m === "number")
      .sort((a, b) => a.distance_m - b.distance_m);

    let maxDiff = 0;
    for (let i = 0; i < validBundles.length - 1; i++) {
      const diff = Math.abs(validBundles[i].distance_m - validBundles[i + 1].distance_m);
      if (diff > maxDiff) maxDiff = diff;
    }

    const mode = validBundles.length <= 1
      ? "nearest_only"
      : maxDiff > 0.2
      ? "nearest_only"
      : "all_bundles_separately";

    const countedBundles =
      mode === "nearest_only" && validBundles.length > 0
        ? [validBundles[0]]
        : validBundles;

    const bundleText = countedBundles
      .map((b, idx) => {
        const count = b.size ?? (Array.isArray(b.rebars) ? b.rebars.length : 0);
        return `B${idx + 1} = ${count} bars`;
      })
      .join(" | ");

    return {
      total_rebars: result?.count ?? 0,
      total_bundles: validBundles.length,
      max_distance_difference: Number(maxDiff.toFixed(2)),
      counting_mode: mode,
      countedBundles,
      bundleText,
    };
  };


  return (
    <div className="page-wrap">
      <div className="dashboard-wrap">
        <span className="card-marker"></span>
        <div className="card">
          <h3 className="gradient-heading" style={{ margin: "0 0 8px 0" }}>
            Upload &amp; Detect
          </h3>
          <p className="muted">
            Upload an image to run the Rebar detection model and save it.
          </p>
          <Upload user={user} onSuccess={handleUploadSuccess} />
        </div>
      </div>

      {/* Gallery Section */}
      <div className="dashboard-wrap">
        <div className="card gallery-card">
          <h4
            className="gradient-heading"
            style={{ margin: "0 0 12px 0", fontSize: "1.1rem" }}
          >
            Recent Uploads &amp; Captures
          </h4>
          {galleryItems && galleryItems.length > 0 ? (
            <div className="shot-grid">
              {galleryItems.map((item, idx) => {
                const { bundles, inBundles, isolated } = parseBundleInfo(
                  item.bundle_info
                );

                // 1) Try to get count from bundle_info: bundles + isolated
                const fromBundle = inBundles + isolated;

                // 2) Fallback to DB count if bundle_info doesn't give anything
                const countFromDb =
                  typeof item.count === "number" ? item.count : 0;

                const displayCount =
                  fromBundle > 0 ? fromBundle : countFromDb;

                const badge = `Count: ${displayCount} | B:${bundles} I:${isolated}`;

                return (
                  <div className="shot-card" key={item.id || idx}>
                    <span className="shot-badge">{badge}</span>
                    {item.image && (
                      <img src={item.image} alt={`Detection ${item.id}`} />
                    )}
                    <div className="shot-footer">{badge}</div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "#d7e2ff" }}>No uploads yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default UploadPage;