// src/pages/ResultPage.jsx
import React, { useEffect, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function ResultPage({ user }) {
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(1);
  const [perPage] = useState(10);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [viewImage, setViewImage] = useState(null);
  const [viewMeta, setViewMeta] = useState(null);

  const userId = user?.id || 1;

  useEffect(() => {
    fetchPage(page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, page]);

  const fetchPage = async (pageNum) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE_URL}/detections?user_id=${userId}&page=${pageNum}&per_page=${perPage}`
      );
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Failed to load history.");
      }
      const data = await res.json();
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message || "Failed to load history.");
    } finally {
      setLoading(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / perPage));

  const handleDelete = async (detId) => {
    if (!window.confirm("Delete this detection?")) return;
    try {
      const res = await fetch(
        `${API_BASE_URL}/detections/${encodeURIComponent(
          detId
        )}?user_id=${userId}`,
        { method: "DELETE" }
      );
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Delete failed.");
      }
      fetchPage(page);
    } catch (err) {
      alert(err.message || "Delete failed.");
    }
  };

  const handleView = async (detId) => {
    try {
      const res = await fetch(
        `${API_BASE_URL}/detections/${encodeURIComponent(
          detId
        )}?user_id=${userId}`
      );
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Failed to load image.");
      }
      const data = await res.json();
      setViewImage(data.image);
      setViewMeta(data.detection);
    } catch (err) {
      alert(err.message || "Failed to load image.");
    }
  };

  const handleExportExcel = () => {
    if (!userId) return;
    window.open(
      `${API_BASE_URL}/detections/export?user_id=${userId}`,
      "_blank"
    );
  };

  const formatTime = (iso) => {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

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
    const bundles = bi.total_bundles || 0;
    const inBundles = bi.rebars_in_bundles || 0;
    const isolated = bi.isolated || bi.total_isolated || 0;
    return { bundles, inBundles, isolated };
  };

  return (
    <div className="dashboard-wrap page-wrap">
      <div className="card gallery-card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 8,
          }}
        >
          <h2 style={{ margin: 0 }}>History</h2>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              className="btn small secondary"
              onClick={handleExportExcel}
            >
              Export Excel
            </button>
          </div>
        </div>

        {error && <div className="alert error">{error}</div>}

        {loading ? (
          <p className="muted">Loading...</p>
        ) : items.length === 0 ? (
          <p className="muted">No detections yet.</p>
        ) : (
          <>
            <div className="table-like">
              <div className="table-row header">
                <div className="th">ID</div>
                <div className="th">Time</div>
                <div className="th">Stream</div>
                <div className="th">Snapshot</div>
                <div className="th">Count</div>
                <div className="th">Bundles</div>
                <div className="th">Isolated</div>
                <div className="th">Actions</div>
              </div>

              {items.map((item, idx) => {
                const { bundles, inBundles, isolated } = parseBundleInfo(
                  item.bundle_info
                );

                // ✅ use DB "count" column for Count
                const displayCount =
                  typeof item.count === "number"
                    ? item.count
                    : bundles > 0
                    ? inBundles
                    : 0;

                // Sequential ID: 1,2,3,... across pages
                const displayId = (page - 1) * perPage + idx + 1;

                return (
                  <div className="table-row" key={item.id}>
                    <div className="td">{displayId}</div>
                    <div className="td ellipsis" title={item.timestamp}>
                      {formatTime(item.timestamp)}
                    </div>
                    <div className="td ellipsis">{item.stream_url}</div>
                    <div className="td ellipsis">
                      {item.thumb_uri ? (
                        <img
                          src={item.thumb_uri}
                          alt="thumb"
                          className="thumb-img"
                        />
                      ) : (
                        item.snapshot_url
                      )}
                    </div>
                    <div className="td">{displayCount}</div>
                    <div className="td">{bundles}</div>
                    <div className="td">{isolated}</div>
                    <div className="td">
                      <button
                        type="button"
                        className="btn small secondary"
                        onClick={() => handleView(item.id)}
                        style={{ marginRight: 6 }}
                      >
                        View
                      </button>
                      <button
                        type="button"
                        className="btn small secondary"
                        onClick={() => handleDelete(item.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="pagination-row">
              <div className="muted">
                Page {page} of {totalPages} · {total} total
              </div>
              <div className="pagination">
                <button
                  type="button"
                  className="btn small secondary"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                >
                  Prev
                </button>
                <div className="page-chip active">{page}</div>
                <button
                  type="button"
                  className="btn small secondary"
                  onClick={() =>
                    setPage((p) => Math.min(totalPages, p + 1))
                  }
                  disabled={page >= totalPages}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {viewImage && (
        <div
          className="view-modal"
          onClick={() => {
            setViewImage(null);
            setViewMeta(null);
          }}
        >
          <div
            className="view-modal-inner"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="view-close"
              type="button"
              onClick={() => {
                setViewImage(null);
                setViewMeta(null);
              }}
            >
              ×
            </button>
            <img src={viewImage} alt="full detection" />
            <div className="view-caption">
              {viewMeta && (
                <>
                  <div>
                    <strong>Detection ID:</strong> {viewMeta.id}
                  </div>
                  <div>
                    <strong>Time:</strong> {formatTime(viewMeta.timestamp)}
                  </div>
                  <div>
                    <strong>Source:</strong> {viewMeta.stream_url} ·{" "}
                    {viewMeta.snapshot_url}
                  </div>
                  <div>
                    <strong>Count (DB field):</strong> {viewMeta.count} ·{" "}
                    <strong>Size:</strong> {viewMeta.width}×
                    {viewMeta.height}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}