// src/pages/UploadPage.jsx
import React, { useEffect, useState } from "react";
import Upload from "../components/Upload";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function UploadPage({ user }) {
  const [galleryItems, setGalleryItems] = useState([]);

  // Fetch recent gallery (limit 10)
  const fetchRecent = async () => {
    if (!user || !user.id) return;
    try {
      const url = new URL(`${API_BASE_URL}/detections/recent`);
      url.searchParams.set("user_id", user.id);
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