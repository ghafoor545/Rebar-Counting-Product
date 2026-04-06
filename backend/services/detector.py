# backend/services/detector.py

import os
import uuid
import base64
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import requests
import onnxruntime as ort
from sklearn.cluster import DBSCAN

from backend.core.config import MODEL_PATH, DET_DIR, THUMB_DIR
from backend.db import get_conn
from backend.utils.utils import utc_now_iso


# ------------------------------------------------------------------
# Bundle Detector Class
# ------------------------------------------------------------------
class RebarBundleDetector:
    def __init__(
        self,
        eps: float = 1.0,  # legacy (kept for backward-compat; NOT used for DBSCAN eps)
        min_bundle_size: int = 3,
        min_samples: int = 2,
        row_tolerance: float = 40.0,
        # NEW: eps computed only from average detected rebar size
        eps_scale: float = 1.25,  # eps = avg_rebar_size * eps_scale
        eps_min: float = 10.0,  # clamp (pixels)
        eps_max: float = 500.0,  # clamp (pixels)
        # legacy flag kept only for compatibility (fixed-eps logic removed)
        use_adaptive_eps: bool = True,
        min_neighbors_in_bundle: Optional[int] = None,
        # segmentation overlay
        draw_seg_masks: bool = True,
        seg_mask_thresh: float = 0.55,
        seg_mask_alpha: float = 0.35,
        draw_mask_contours: bool = True,
        mask_contour_thickness: int = 2,
        # depth filtering (OAK-D Pro)
        use_depth_filter: bool = True,
        max_detection_distance_mm: float = 3000.0,
        min_detection_distance_mm: float = 200.0,
        # Bundle distance tracking
        track_bundle_distances: bool = True,
        nearest_bundle_only: bool = False,
        # debug
        debug=False,
    ):
        # Legacy (not used anymore for DBSCAN eps)
        self.eps = eps

        self.min_bundle_size = min_bundle_size
        self.min_samples = min_samples
        self.row_tolerance = row_tolerance

        # NEW dynamic eps config
        self.eps_scale = float(eps_scale)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)

        # kept for compatibility (no longer enables/disables fixed eps)
        self.use_adaptive_eps = use_adaptive_eps
        self.min_neighbors_in_bundle = min_neighbors_in_bundle

        self.draw_seg_masks = draw_seg_masks
        self.seg_mask_thresh = seg_mask_thresh
        self.seg_mask_alpha = seg_mask_alpha
        self.draw_mask_contours = draw_mask_contours
        self.mask_contour_thickness = mask_contour_thickness

        # Depth filter settings
        self.use_depth_filter = use_depth_filter
        self.max_detection_distance_mm = max_detection_distance_mm
        self.min_detection_distance_mm = min_detection_distance_mm

        # Bundle distance tracking
        self.track_bundle_distances = track_bundle_distances
        self.nearest_bundle_only = nearest_bundle_only

        self.debug = debug

    # ------------------------------------------------------------------
    # JSON safety
    # ------------------------------------------------------------------
    def _make_json_safe(self, obj):
        """Recursively convert non-JSON-serializable values to None"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return self._make_json_safe(obj.tolist())
        if isinstance(obj, (float, int)):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
        return obj

    # ------------------------------------------------------------------
    # Dynamic eps (only from avg rebar size)
    # ------------------------------------------------------------------
    def _dynamic_eps_from_sizes(self, sizes: np.ndarray) -> float:
        """
        Compute DBSCAN eps purely from average detected rebar size (box diagonal).
        No fixed eps baseline and no median-distance logic.
        """
        if sizes is None or len(sizes) == 0:
            return float(self.eps_min)

        avg_size = float(np.mean(sizes))
        eps = avg_size * float(self.eps_scale)
        eps = float(np.clip(eps, self.eps_min, self.eps_max))

        if self.debug:
            print(
                f"[DBSCAN] avg_rebar_size={avg_size:.2f}, eps_scale={self.eps_scale:.2f} "
                f"=> eps={eps:.2f} (clamp={self.eps_min:.1f}..{self.eps_max:.1f})"
            )
        return eps

    # kept for backward compatibility
    def _calculate_adaptive_eps(self, centers: np.ndarray, sizes: np.ndarray) -> float:
        return self._dynamic_eps_from_sizes(sizes)

    # ------------------------------------------------------------------
    # Distance per bundle (median of members)
    # ------------------------------------------------------------------
    def _calculate_bundle_distance(
        self,
        bundle: Dict[str, Any],
        depth_map: np.ndarray,
        boxes: np.ndarray,
    ) -> Optional[float]:
        if depth_map is None or not self.track_bundle_distances:
            return None

        try:
            from backend.services.oak_utils import get_depth_at_point
        except ImportError:
            return None

        distances = []
        for rebar in bundle["rebars"]:
            box = rebar["box"]
            x1, y1, x2, y2 = map(int, box[:4])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            depth_mm = get_depth_at_point(depth_map, cx, cy, sample_radius=5)
            if depth_mm is not None and depth_mm > 0:
                distances.append(depth_mm)

        if distances:
            bundle_distance = float(np.median(distances))
            bundle["distance_mm"] = bundle_distance
            bundle["distance_m"] = round(bundle_distance / 1000.0, 2)

            if self.debug:
                print(
                    f"[Bundle {bundle['bundle_id']}] Distance: {bundle_distance:.0f}mm "
                    f"({bundle_distance/1000:.2f}m) from {len(distances)} rebars"
                )
            return bundle_distance

        bundle["distance_mm"] = None
        bundle["distance_m"] = None
        return None

    # ------------------------------------------------------------------
    # Sorting helpers
    # ------------------------------------------------------------------
    def _sort_rebars_top_to_bottom_by_indices(self, indices, centers):
        """Sort rebars from top to bottom by y-coordinate (smaller y = top)"""
        if len(indices) <= 1:
            return indices
        pairs = list(zip(indices, centers))
        pairs.sort(key=lambda x: x[1][1])
        return np.array([idx for idx, _ in pairs])

    # ------------------------------------------------------------------
    # Bundle clustering (DBSCAN)
    # ------------------------------------------------------------------
    def detect_bundles(self, boxes: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if len(boxes) == 0:
            return self._make_json_safe(
                {
                    "bundles": [],
                    "isolated_rebars": [],
                    "total_bundles": 0,
                    "total_rebars_in_bundles": 0,
                    "total_isolated": 0,
                    "total_count": 0,
                    "id_mapping": {},
                    "display_summary": {"total": 0, "bundles": 0, "rebars_in_bundles": 0, "isolated": 0},
                    "nearest_bundle": None,
                }
            )

        centers, sizes = self._get_box_centers_and_sizes(boxes)

        # eps is ALWAYS dynamic (avg size only)
        eps_to_use = self._dynamic_eps_from_sizes(sizes)

        # More inclusive DBSCAN settings (kept from your original)
        min_samples_for_dbscan = max(1, int(self.min_samples) - 1)
        clustering = DBSCAN(eps=eps_to_use, min_samples=min_samples_for_dbscan).fit(centers)
        labels = clustering.labels_

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        # More inclusive bundle criteria (kept from your original)
        min_bundle_size_effective = max(2, int(self.min_bundle_size) - 1)
        bundle_clusters = [
            label
            for label in set(labels.tolist())
            if label != -1 and cluster_sizes.get(label, 0) >= min_bundle_size_effective
        ]

        bundles: List[Dict[str, Any]] = []
        isolated_rebars: List[Dict[str, Any]] = []
        id_mapping: Dict[int, Dict[str, Any]] = {}
        bundle_counter = 0

        unassigned = set(range(len(boxes)))

        min_neighbors = (
            max(1, (int(self.min_samples) - 2))
            if self.min_neighbors_in_bundle is None
            else int(self.min_neighbors_in_bundle)
        )

        for label in bundle_clusters:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_centers = centers[cluster_indices]

            # neighbor counts inside the cluster
            diff = cluster_centers[:, None, :] - cluster_centers[None, :, :]
            dist = np.sqrt((diff**2).sum(axis=2))
            neighbor_counts = (dist <= eps_to_use).sum(axis=1) - 1

            keep_local = neighbor_counts >= min_neighbors
            kept_indices = cluster_indices[keep_local]

            # keep small ones anyway (original behavior)
            if len(kept_indices) < 2:
                kept_indices = cluster_indices

            bundle_counter += 1
            kept_centers = centers[kept_indices]
            kept_boxes = boxes[kept_indices]
            cluster_size = int(len(kept_indices))

            sorted_indices = self._sort_rebars_top_to_bottom_by_indices(kept_indices, kept_centers)

            bundle_rebars = []
            for bundle_idx, global_idx in enumerate(sorted_indices, start=1):
                bundle_rebars.append(
                    {
                        "global_index": int(global_idx),
                        "bundle_index": int(bundle_idx),
                        "display_id": int(bundle_idx),
                        "box": boxes[global_idx].tolist(),
                        "center": centers[global_idx].tolist(),
                        "row": self._get_row_number(centers[global_idx], kept_centers),
                    }
                )
                id_mapping[int(global_idx)] = {
                    "display_id": int(bundle_idx),
                    "bundle_id": int(bundle_counter),
                    "bundle_index": int(bundle_idx),
                    "type": "bundle",
                    "group_size": int(cluster_size),
                }

            all_x1 = kept_boxes[:, 0]
            all_y1 = kept_boxes[:, 1]
            all_x2 = kept_boxes[:, 2]
            all_y2 = kept_boxes[:, 3]

            bundle_dict = {
                "bundle_id": int(bundle_counter),
                "size": int(cluster_size),
                "rebars": bundle_rebars,
                "global_indices": sorted_indices.tolist(),
                "bounds": [
                    float(np.min(all_x1)),
                    float(np.min(all_y1)),
                    float(np.max(all_x2)),
                    float(np.max(all_y2)),
                ],
                "distance_mm": None,
                "distance_m": None,
                "is_nearest": False,
                "counted": True,
            }

            if depth_map is not None and self.track_bundle_distances:
                self._calculate_bundle_distance(bundle_dict, depth_map, boxes)

            bundles.append(bundle_dict)

            for gi in kept_indices.tolist():
                unassigned.discard(int(gi))

        # isolated rebars
        isolated_indices = sorted(list(unassigned))
        if isolated_indices:
            isolated_centers = centers[isolated_indices]
            sorted_isolated = self._sort_rebars_top_to_bottom_by_indices(
                np.array(isolated_indices), isolated_centers
            )
            for display_idx, global_idx in enumerate(sorted_isolated, start=1):
                isolated_rebars.append(
                    {
                        "global_index": int(global_idx),
                        "display_id": int(display_idx),
                        "box": boxes[global_idx].tolist(),
                        "center": centers[global_idx].tolist(),
                        "type": "isolated",
                        "group_size": 1,
                    }
                )
                id_mapping[int(global_idx)] = {
                    "display_id": int(display_idx),
                    "type": "isolated",
                    "group_size": 1,
                }

        # nearest bundle info (normal mode)
        nearest_bundle_info = None
        if bundles and self.track_bundle_distances:
            valid_bundles = [b for b in bundles if b.get("distance_mm") is not None]
            if valid_bundles:
                nearest = min(valid_bundles, key=lambda b: b["distance_mm"])
                nearest_bundle_info = {
                    "bundle_id": nearest["bundle_id"],
                    "distance_mm": nearest["distance_mm"],
                    "distance_m": nearest["distance_m"],
                    "size": nearest["size"],
                }
                for b in bundles:
                    b["is_nearest"] = (b["bundle_id"] == nearest["bundle_id"])

        # nearest-only mode (optional)
        if self.nearest_bundle_only and bundles:
            valid_bundles = [b for b in bundles if b.get("distance_mm") is not None]
            if valid_bundles:
                nearest_bundle = min(valid_bundles, key=lambda b: b["distance_mm"])
                for b in bundles:
                    b["is_nearest"] = (b["bundle_id"] == nearest_bundle["bundle_id"])
                    b["counted"] = (b["bundle_id"] == nearest_bundle["bundle_id"])
                bundles = [nearest_bundle]

                nearest_bundle_info = {
                    "bundle_id": nearest_bundle["bundle_id"],
                    "distance_mm": nearest_bundle["distance_mm"],
                    "distance_m": nearest_bundle["distance_m"],
                    "size": nearest_bundle["size"],
                }

        result = {
            "bundles": bundles,
            "isolated_rebars": isolated_rebars,
            "total_bundles": int(len(bundles)),
            "total_rebars_in_bundles": int(sum(b["size"] for b in bundles)),
            "total_isolated": int(len(isolated_rebars)),
            "total_count": int(len(boxes)),
            "id_mapping": id_mapping,
            "display_summary": {
                "total": int(len(boxes)),
                "bundles": int(len(bundles)),
                "rebars_in_bundles": int(sum(b["size"] for b in bundles)),
                "isolated": int(len(isolated_rebars)),
            },
            "nearest_bundle": nearest_bundle_info,
        }
        return self._make_json_safe(result)

    def _get_row_number(self, center, all_centers):
        y_coords = sorted(set([float(c[1]) for c in all_centers]))
        for i, y in enumerate(y_coords):
            if abs(float(center[1]) - y) <= self.row_tolerance:
                return i + 1
        return 0

    def _get_box_centers_and_sizes(self, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centers = []
        sizes = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append([cx, cy])
            size = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            sizes.append(size)
        return np.array(centers, dtype=np.float32), np.array(sizes, dtype=np.float32)

    # ------------------------------------------------------------------
    # Depth filter (UPDATED: keep unknown depth to avoid dropping small rebars)
    # ------------------------------------------------------------------
    def filter_by_depth(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        coeffs: Optional[np.ndarray],
        depth_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if len(boxes) == 0 or depth_map is None:
            return boxes, scores, coeffs

        try:
            from backend.services.oak_utils import get_depth_at_point
        except ImportError:
            return boxes, scores, coeffs

        keep_mask = np.zeros(len(boxes), dtype=bool)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Larger radius helps tiny objects
            depth_mm = get_depth_at_point(depth_map, cx, cy, sample_radius=9)

            # KEEP if depth is unknown (common depth holes on small objects)
            if depth_mm is None:
                keep_mask[i] = True
            else:
                keep_mask[i] = (self.min_detection_distance_mm <= depth_mm <= self.max_detection_distance_mm)

        filtered_boxes = boxes[keep_mask]
        filtered_scores = scores[keep_mask]
        filtered_coeffs = coeffs[keep_mask] if coeffs is not None else None

        if self.debug:
            print(f"[Depth Filter] Kept {len(filtered_boxes)}/{len(boxes)} boxes")

        return filtered_boxes, filtered_scores, filtered_coeffs

    # ------------------------------------------------------------------
    # Drawing: annotate counted bundles (bundle-local 1..N)
    # ------------------------------------------------------------------
    def annotate_counted_bundles(self, image_bgr, boxes, bundle_info, counted_bundle_ids):
        img = image_bgr.copy()
        if not bundle_info:
            return img

        bundles = bundle_info.get("bundles", [])
        for bundle in bundles:
            bundle_id = bundle.get("bundle_id")
            if bundle_id not in counted_bundle_ids:
                continue

            rebars = bundle.get("rebars", []) or []
            if not rebars:
                continue

            sorted_rebars = sorted(
                rebars,
                key=lambda r: (
                    ((r.get("box", [0, 0, 0, 0])[1] + r.get("box", [0, 0, 0, 0])[3]) / 2.0),
                    ((r.get("box", [0, 0, 0, 0])[0] + r.get("box", [0, 0, 0, 0])[2]) / 2.0),
                ),
            )

            for idx, rebar in enumerate(sorted_rebars, start=1):
                box = rebar.get("box")
                if not box or len(box) < 4:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                display_txt = str(idx)
                font = cv2.FONT_HERSHEY_SIMPLEX

                box_size = max(1, min(abs(x2 - x1), abs(y2 - y1)))
                font_scale = max(0.16, min(0.4, box_size / 140.0))
                thickness = 1

                (text_w, text_h), _ = cv2.getTextSize(display_txt, font, font_scale, thickness)
                text_x = cx - text_w // 2
                text_y = cy + text_h // 2

                cv2.putText(
                    img, display_txt, (text_x, text_y), font, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA
                )
                cv2.putText(
                    img, display_txt, (text_x, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA
                )

        return img

    # ------------------------------------------------------------------
    # Snapshot fetch helper
    # ------------------------------------------------------------------
    def fetch_snapshot(self, url: str):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                return None, f"Failed to fetch snapshot: HTTP {r.status_code}"
            nparr = np.frombuffer(r.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return None, "Failed to decode image from snapshot."
            return image, None
        except Exception as e:
            return None, f"Snapshot fetch error: {e}"

    # ------------------------------------------------------------------
    # Resize helpers
    # ------------------------------------------------------------------
    def to_hd_1080p(self, image_bgr, background=(18, 24, 31)):
        target_w, target_h = 1920, 1080
        h, w = image_bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), background, dtype=np.uint8)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (left, top)

    # ------------------------------------------------------------------
    # Box helpers
    # ------------------------------------------------------------------
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def scale_boxes(self, boxes, r, dwdh, orig_shape):
        left, top = dwdh
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= float(left)
        boxes[:, [1, 3]] -= float(top)
        boxes[:, :4] /= float(r)
        h, w = orig_shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    def iou_one_to_many(self, box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1]) + 1e-6
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-6
        return inter / (area1 + area2 - inter + 1e-6)

    def nms(self, boxes, scores, iou_thres=0.45, max_det=10000):
        if len(boxes) == 0:
            return []
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if len(keep) >= max_det or idxs.size == 1:
                break
            ious = self.iou_one_to_many(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_thres]
        return keep

    # ------------------------------------------------------------------
    # Activation + output parsing
    # ------------------------------------------------------------------
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def maybe_sigmoid(self, arr):
        a = np.asarray(arr)
        if a.size == 0:
            return a
        mn, mx = float(np.min(a)), float(np.max(a))
        if 0.0 <= mn and mx <= 1.0:
            return a
        return self.sigmoid(a)

    def parse_onnx_outputs(self, outs):
        arrs = [np.asarray(o) for o in outs]

        # Segmentation outputs: [1, N, no] and [1, nm, mh, mw]
        if len(arrs) == 2 and arrs[0].ndim == 3 and arrs[1].ndim == 4:
            return "seg", (arrs[0], arrs[1])

        # Some exports output [N,6] or [1,N,6]
        for a in arrs:
            if a.ndim == 3 and a.shape[-1] == 6:
                return "nms", (a[0] if a.shape[0] == 1 else a)
            if a.ndim == 2 and a.shape[-1] == 6:
                return "nms", a

        # Raw outputs
        z = arrs[0]
        if z.ndim == 3:
            if z.shape[1] < z.shape[2]:
                z = np.transpose(z, (0, 2, 1))
            z = z[0]
        elif z.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected ONNX output shape: {z.shape}")

        return "raw", z

    def preprocess_for_onnx(self, image_bgr, in_hw):
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise ValueError("preprocess_for_onnx: empty image (None)")

        in_h, in_w = in_hw
        lb, r, dwdh = self.letterbox(image_bgr, (in_h, in_w))
        img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img, r, dwdh

    # ------------------------------------------------------------------
    # Segmentation masks
    # ------------------------------------------------------------------
    def decode_seg_masks(
        self,
        coeffs: np.ndarray,
        proto: np.ndarray,
        boxes_xyxy_orig: np.ndarray,
        r: float,
        dwdh: Tuple[float, float],
        orig_shape,
        in_hw: Tuple[int, int],
        mask_thresh: float = 0.5,
    ) -> List[np.ndarray]:
        if coeffs is None or len(coeffs) == 0:
            return []

        in_h, in_w = in_hw
        orig_h, orig_w = orig_shape[:2]
        left, top = map(int, dwdh)

        proto = proto.astype(np.float32)
        coeffs = coeffs.astype(np.float32)

        nm, mh, mw = proto.shape
        proto_flat = proto.reshape(nm, -1)
        masks = self.sigmoid(coeffs @ proto_flat).reshape(-1, mh, mw)

        new_w = int(round(orig_w * float(r)))
        new_h = int(round(orig_h * float(r)))

        out_masks: List[np.ndarray] = []
        for i, m in enumerate(masks):
            m_in = cv2.resize(m, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

            x1p = max(left, 0)
            y1p = max(top, 0)
            x2p = min(left + new_w, in_w)
            y2p = min(top + new_h, in_h)
            if x2p <= x1p or y2p <= y1p:
                out_masks.append(np.zeros((orig_h, orig_w), dtype=bool))
                continue

            m_crop = m_in[y1p:y2p, x1p:x2p]
            m_orig = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            m_bin = (m_orig >= float(mask_thresh))

            bx1, by1, bx2, by2 = boxes_xyxy_orig[i]
            x1 = int(max(0, np.floor(bx1)))
            y1 = int(max(0, np.floor(by1)))
            x2 = int(min(orig_w, np.ceil(bx2)))
            y2 = int(min(orig_h, np.ceil(by2)))

            if y1 > 0:
                m_bin[:y1, :] = False
            if y2 < orig_h:
                m_bin[y2:, :] = False
            if x1 > 0:
                m_bin[:, :x1] = False
            if x2 < orig_w:
                m_bin[:, x2:] = False

            out_masks.append(m_bin)

        return out_masks

    def _color_for_index(self, i: int) -> Tuple[int, int, int]:
        h = int((i * 37) % 180)
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _overlay_instance_masks(
        self,
        image_bgr: np.ndarray,
        masks: List[np.ndarray],
        colors: List[Tuple[int, int, int]],
        alpha: float,
        draw_contours: bool,
        contour_thickness: int,
    ) -> np.ndarray:
        out = image_bgr.copy()

        for i, m in enumerate(masks):
            if m is None:
                continue
            if m.dtype != np.bool_:
                m = m.astype(bool)
            if not np.any(m):
                continue

            color = colors[i]
            out[m] = (
                out[m].astype(np.float32) * (1.0 - float(alpha))
                + np.array(color, dtype=np.float32) * float(alpha)
            ).astype(np.uint8)

            if draw_contours:
                mu8 = (m.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(out, contours, -1, color, int(contour_thickness))

        return out

    # ------------------------------------------------------------------
    # Core detect (single pass, whole frame) -> returns RGB image + bundle_info
    # ------------------------------------------------------------------
    def detect_rebars(
        self,
        image_bgr,
        model,
        depth_map: Optional[np.ndarray] = None,
        class_id: int = 0,
        conf: float = 0.5,
        iou: float = 0.7,
        max_det: int = 10000,
    ):
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return None, 0, "Empty image (None) passed to detect_rebars()", None

        try:
            sess = model["sess"]
            in_name = model["in_name"]
            in_hw = model["in_hw"]
            in_h, in_w = in_hw

            blob, r, dwdh = self.preprocess_for_onnx(image_bgr, in_hw)
            outs = sess.run(None, {in_name: blob})
            kind, data = self.parse_onnx_outputs(outs)

            dets_xyxy = []
            seg_masks: Optional[List[np.ndarray]] = None
            instance_colors: Optional[List[Tuple[int, int, int]]] = None

            # Segmentation branch
            if kind == "seg":
                pred_raw, proto_raw = data
                pred = pred_raw[0]
                proto = proto_raw[0]

                if pred.ndim != 2:
                    raise ValueError(f"Unexpected seg pred ndim={pred.ndim}")
                if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
                    pred = pred.T

                pred = pred.astype(np.float32)
                no = int(pred.shape[1])
                nm = int(proto.shape[0])

                nc = no - 4 - nm
                if nc <= 0:
                    nc = 1

                boxes = pred[:, 0:4].copy()

                scores_mat = self.maybe_sigmoid(pred[:, 4 : 4 + nc])
                if scores_mat.ndim == 1:
                    scores_mat = scores_mat[:, None]
                if int(class_id) >= scores_mat.shape[1]:
                    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0, "class_id out of range", None
                scores = scores_mat[:, int(class_id)]

                coeff_start = 4 + nc
                coeff_end = coeff_start + nm
                if coeff_end > no:
                    coeff_start = 5
                    coeff_end = 5 + nm
                coeffs = pred[:, coeff_start:coeff_end].copy()
                if coeffs.shape[1] != nm:
                    raise ValueError(f"Bad coeffs shape {coeffs.shape}")

                if np.nanmax(boxes) <= 1.5:
                    boxes[:, [0, 2]] *= float(in_w)
                    boxes[:, [1, 3]] *= float(in_h)

                b_xyxy_from_xywh = self.xywh2xyxy(boxes.copy())
                valid1 = (
                    (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0])
                    & (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
                ).sum()
                b_xyxy_as_is = boxes.copy()
                valid2 = (
                    (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0])
                    & (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
                ).sum()
                b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

                keep_mask = scores >= float(conf)
                if np.any(keep_mask):
                    b_xyxy = b_xyxy[keep_mask]
                    s = scores[keep_mask]
                    coeffs = coeffs[keep_mask]

                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    if len(s) > max_det:
                        topk = np.argsort(-s)[:max_det]
                        b_xyxy = b_xyxy[topk]
                        s = s[topk]
                        coeffs = coeffs[topk]

                    keep = self.nms(b_xyxy, s, iou_thres=float(iou), max_det=max_det)
                    if keep:
                        b_xyxy = b_xyxy[keep]
                        s = s[keep]
                        coeffs = coeffs[keep]

                        if self.use_depth_filter and depth_map is not None:
                            b_xyxy, s, coeffs = self.filter_by_depth(b_xyxy, s, coeffs, depth_map)

                        if len(b_xyxy) > 0:
                            dets_xyxy = b_xyxy.tolist()

                            if self.draw_seg_masks:
                                seg_masks = self.decode_seg_masks(
                                    coeffs=coeffs,
                                    proto=proto,
                                    boxes_xyxy_orig=b_xyxy,
                                    r=float(r),
                                    dwdh=dwdh,
                                    orig_shape=image_bgr.shape,
                                    in_hw=in_hw,
                                    mask_thresh=float(self.seg_mask_thresh),
                                )
                                instance_colors = [
                                    self._color_for_index(i) for i in range(len(seg_masks))
                                ]

            # Detection (NMS-like) branch
            elif kind == "nms":
                d = data.reshape(-1, 6).astype(np.float32)
                cls = np.round(d[:, 5]).astype(np.int32)
                scr = d[:, 4]
                mask = (scr >= conf) & (cls == int(class_id))
                if np.any(mask):
                    b = d[mask, :4]
                    s = scr[mask]
                    if np.nanmax(b) <= 1.5:
                        b[:, [0, 2]] *= float(in_w)
                        b[:, [1, 3]] *= float(in_h)
                    b = self.scale_boxes(b, r, dwdh, image_bgr.shape)

                    if self.use_depth_filter and depth_map is not None:
                        b, s, _ = self.filter_by_depth(b, s, None, depth_map)

                    if len(b) > 0:
                        dets_xyxy = b.tolist()

            # Detection (raw) branch
            else:
                z = data
                C = z.shape[1]
                boxes = z[:, :4].astype(np.float32)

                scores_mat_a = None
                scores_mat_b = None

                if C - 4 > 0:
                    scores_mat_a = self.maybe_sigmoid(z[:, 4:])
                if C - 5 > 0:
                    obj = self.maybe_sigmoid(z[:, 4:5])
                    cls_b = self.maybe_sigmoid(z[:, 5:])
                    scores_mat_b = obj * cls_b

                def count_above(m, t=0.25):
                    return int((m is not None) and (m.max(axis=1) >= t).sum())

                cnt_a = count_above(scores_mat_a)
                cnt_b = count_above(scores_mat_b)

                scores_mat = (
                    scores_mat_a
                    if (scores_mat_a is not None and (scores_mat_b is None or cnt_a >= cnt_b))
                    else scores_mat_b
                )

                if scores_mat is None or scores_mat.shape[1] <= class_id:
                    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0, "Model scores not found.", None

                scores = scores_mat[:, class_id]

                if np.nanmax(boxes) <= 1.5:
                    boxes[:, [0, 2]] *= float(in_w)
                    boxes[:, [1, 3]] *= float(in_h)

                b_xyxy_from_xywh = self.xywh2xyxy(boxes.copy())
                valid1 = (
                    (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0])
                    & (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
                ).sum()
                b_xyxy_as_is = boxes.copy()
                valid2 = (
                    (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0])
                    & (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
                ).sum()
                b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

                keep_mask = scores >= float(conf)
                if np.any(keep_mask):
                    b_xyxy = b_xyxy[keep_mask]
                    s = scores[keep_mask]

                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    if len(s) > max_det:
                        topk = np.argsort(-s)[:max_det]
                        b_xyxy = b_xyxy[topk]
                        s = s[topk]

                    keep = self.nms(b_xyxy, s, iou_thres=float(iou), max_det=max_det)
                    if keep:
                        b_xyxy = b_xyxy[keep]
                        s = s[keep]

                        if self.use_depth_filter and depth_map is not None:
                            b_xyxy, s, _ = self.filter_by_depth(b_xyxy, s, None, depth_map)

                        if len(b_xyxy) > 0:
                            dets_xyxy = b_xyxy.tolist()

            # Build bundle info
            if dets_xyxy:
                boxes_array = np.array(dets_xyxy, dtype=np.float32)

                annotated = image_bgr.copy()
                if seg_masks and instance_colors:
                    annotated = self._overlay_instance_masks(
                        annotated,
                        seg_masks,
                        instance_colors,
                        alpha=float(self.seg_mask_alpha),
                        draw_contours=bool(self.draw_mask_contours),
                        contour_thickness=int(self.mask_contour_thickness),
                    )

                bundle_info = self.detect_bundles(boxes_array, depth_map)
                count = len(boxes_array)
            else:
                annotated = image_bgr.copy()
                bundle_info = None
                count = 0

            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), int(count), None, bundle_info

        except Exception as e:
            try:
                fallback = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                fallback = None
            return fallback, 0, f"Detection error: {e}", None

    # ------------------------------------------------------------------
    # TILING / CROP-ZOOM INFERENCE (boxes-only; does NOT merge seg masks)
    # ------------------------------------------------------------------
    def _iter_tiles(self, w: int, h: int, tile_size: int, overlap: float):
        """Yield (x0,y0,x1,y1) tiles covering the image with overlap."""
        tile_size = int(tile_size)
        if tile_size <= 0:
            tile_size = min(w, h)

        if tile_size >= w and tile_size >= h:
            yield (0, 0, w, h)
            return

        overlap = float(overlap)
        overlap = max(0.0, min(0.95, overlap))

        step = int(tile_size * (1.0 - overlap))
        step = max(1, step)

        max_x0 = max(0, w - tile_size)
        max_y0 = max(0, h - tile_size)

        xs = list(range(0, max_x0 + 1, step)) or [0]
        ys = list(range(0, max_y0 + 1, step)) or [0]

        if xs[-1] != max_x0:
            xs.append(max_x0)
        if ys[-1] != max_y0:
            ys.append(max_y0)

        for y0 in ys:
            for x0 in xs:
                x1 = min(w, x0 + tile_size)
                y1 = min(h, y0 + tile_size)
                yield (x0, y0, x1, y1)

    def _infer_boxes_single(
        self,
        image_bgr: np.ndarray,
        model,
        depth_map: Optional[np.ndarray] = None,
        class_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 10000,
        apply_depth_filter: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Run model on a single image and return (boxes_xyxy, scores, error).
        boxes are in this image coordinate space.
        NOTE: seg masks are not returned here (tiling is box-only).
        """
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), "Empty image"

        try:
            sess = model["sess"]
            in_name = model["in_name"]
            in_hw = model["in_hw"]
            in_h, in_w = in_hw

            blob, r, dwdh = self.preprocess_for_onnx(image_bgr, in_hw)
            outs = sess.run(None, {in_name: blob})
            kind, data = self.parse_onnx_outputs(outs)

            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            coeffs = None  # only for depth filter signature

            if kind == "seg":
                pred_raw, proto_raw = data
                pred = pred_raw[0]
                proto = proto_raw[0]

                if pred.ndim != 2:
                    raise ValueError(f"Unexpected seg pred ndim={pred.ndim}")
                if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
                    pred = pred.T

                pred = pred.astype(np.float32)
                no = int(pred.shape[1])
                nm = int(proto.shape[0])

                nc = no - 4 - nm
                if nc <= 0:
                    nc = 1

                boxes = pred[:, 0:4].copy()

                scores_mat = self.maybe_sigmoid(pred[:, 4 : 4 + nc])
                if scores_mat.ndim == 1:
                    scores_mat = scores_mat[:, None]
                if int(class_id) >= scores_mat.shape[1]:
                    return boxes_xyxy, scores, "class_id out of range"
                scr = scores_mat[:, int(class_id)]

                coeff_start = 4 + nc
                coeff_end = coeff_start + nm
                if coeff_end > no:
                    coeff_start = 5
                    coeff_end = 5 + nm
                coeffs = pred[:, coeff_start:coeff_end].copy()

                if np.nanmax(boxes) <= 1.5:
                    boxes[:, [0, 2]] *= float(in_w)
                    boxes[:, [1, 3]] *= float(in_h)

                b_xyxy_from_xywh = self.xywh2xyxy(boxes.copy())
                valid1 = (
                    (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0])
                    & (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
                ).sum()
                b_xyxy_as_is = boxes.copy()
                valid2 = (
                    (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0])
                    & (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
                ).sum()
                b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

                keep = scr >= float(conf)
                if np.any(keep):
                    b_xyxy = b_xyxy[keep]
                    s = scr[keep].astype(np.float32)
                    coeffs = coeffs[keep]

                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    if len(s) > max_det:
                        topk = np.argsort(-s)[:max_det]
                        b_xyxy = b_xyxy[topk]
                        s = s[topk]
                        coeffs = coeffs[topk]

                    keep_idx = self.nms(b_xyxy, s, iou_thres=float(iou), max_det=max_det)
                    if keep_idx:
                        b_xyxy = b_xyxy[keep_idx]
                        s = s[keep_idx]
                        coeffs = coeffs[keep_idx]

                        if apply_depth_filter and self.use_depth_filter and depth_map is not None:
                            b_xyxy, s, coeffs = self.filter_by_depth(b_xyxy, s, coeffs, depth_map)

                        boxes_xyxy = b_xyxy.astype(np.float32)
                        scores = s.astype(np.float32)

            elif kind == "nms":
                d = data.reshape(-1, 6).astype(np.float32)
                cls = np.round(d[:, 5]).astype(np.int32)
                scr = d[:, 4]
                mask = (scr >= float(conf)) & (cls == int(class_id))
                if np.any(mask):
                    b = d[mask, :4].astype(np.float32)
                    s = scr[mask].astype(np.float32)

                    if np.nanmax(b) <= 1.5:
                        b[:, [0, 2]] *= float(in_w)
                        b[:, [1, 3]] *= float(in_h)

                    b = self.scale_boxes(b, r, dwdh, image_bgr.shape)

                    if len(s) > max_det:
                        topk = np.argsort(-s)[:max_det]
                        b = b[topk]
                        s = s[topk]

                    keep_idx = self.nms(b, s, iou_thres=float(iou), max_det=max_det)
                    if keep_idx:
                        b = b[keep_idx]
                        s = s[keep_idx]

                        if apply_depth_filter and self.use_depth_filter and depth_map is not None:
                            b, s, _ = self.filter_by_depth(b, s, None, depth_map)

                        boxes_xyxy = b.astype(np.float32)
                        scores = s.astype(np.float32)

            else:
                z = data
                C = z.shape[1]
                boxes = z[:, :4].astype(np.float32)

                scores_mat_a = None
                scores_mat_b = None

                if C - 4 > 0:
                    scores_mat_a = self.maybe_sigmoid(z[:, 4:])
                if C - 5 > 0:
                    obj = self.maybe_sigmoid(z[:, 4:5])
                    cls_b = self.maybe_sigmoid(z[:, 5:])
                    scores_mat_b = obj * cls_b

                def count_above(m, t=0.25):
                    return int((m is not None) and (m.max(axis=1) >= t).sum())

                cnt_a = count_above(scores_mat_a)
                cnt_b = count_above(scores_mat_b)

                scores_mat = (
                    scores_mat_a
                    if (scores_mat_a is not None and (scores_mat_b is None or cnt_a >= cnt_b))
                    else scores_mat_b
                )

                if scores_mat is None or scores_mat.shape[1] <= int(class_id):
                    return boxes_xyxy, scores, "Model scores not found"

                scr = scores_mat[:, int(class_id)].astype(np.float32)

                if np.nanmax(boxes) <= 1.5:
                    boxes[:, [0, 2]] *= float(in_w)
                    boxes[:, [1, 3]] *= float(in_h)

                b_xyxy_from_xywh = self.xywh2xyxy(boxes.copy())
                valid1 = (
                    (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0])
                    & (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
                ).sum()
                b_xyxy_as_is = boxes.copy()
                valid2 = (
                    (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0])
                    & (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
                ).sum()
                b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

                keep = scr >= float(conf)
                if np.any(keep):
                    b_xyxy = b_xyxy[keep]
                    s = scr[keep]

                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    if len(s) > max_det:
                        topk = np.argsort(-s)[:max_det]
                        b_xyxy = b_xyxy[topk]
                        s = s[topk]

                    keep_idx = self.nms(b_xyxy, s, iou_thres=float(iou), max_det=max_det)
                    if keep_idx:
                        b_xyxy = b_xyxy[keep_idx]
                        s = s[keep_idx]

                        if apply_depth_filter and self.use_depth_filter and depth_map is not None:
                            b_xyxy, s, _ = self.filter_by_depth(b_xyxy, s, None, depth_map)

                        boxes_xyxy = b_xyxy.astype(np.float32)
                        scores = s.astype(np.float32)

            return boxes_xyxy, scores, None

        except Exception as e:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), f"infer error: {e}"

    def _infer_boxes_tiled(
        self,
        image_bgr: np.ndarray,
        model,
        depth_map: Optional[np.ndarray] = None,
        tile_size: int = 640,
        overlap: float = 0.25,
        class_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 10000,
        apply_depth_filter: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """
        Tiled inference: run detection on overlapping tiles, merge, global NMS.
        Returns boxes in full-image coordinates.
        """
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), "Empty image"

        h, w = image_bgr.shape[:2]

        all_boxes = []
        all_scores = []

        # Depth filter applied once globally after merging (more stable for small objects)
        for (x0, y0, x1, y1) in self._iter_tiles(w, h, tile_size=tile_size, overlap=overlap):
            tile = image_bgr[y0:y1, x0:x1]

            b, s, err = self._infer_boxes_single(
                tile,
                model=model,
                depth_map=None,
                class_id=class_id,
                conf=conf,
                iou=iou,
                max_det=max_det,
                apply_depth_filter=False,
            )
            if err or b is None or len(b) == 0:
                continue

            b = b.copy()
            b[:, [0, 2]] += float(x0)
            b[:, [1, 3]] += float(y0)

            all_boxes.append(b)
            all_scores.append(s)

        if not all_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), None

        boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(all_scores, axis=0).astype(np.float32)

        keep_idx = self.nms(boxes, scores, iou_thres=float(iou), max_det=max_det)
        if not keep_idx:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), None

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        if apply_depth_filter and self.use_depth_filter and depth_map is not None and len(boxes) > 0:
            boxes, scores, _ = self.filter_by_depth(boxes, scores, None, depth_map)

        return boxes, scores, None

    def detect_rebars_tiled(
        self,
        image_bgr: np.ndarray,
        model,
        depth_map: Optional[np.ndarray] = None,
        tile_size: int = 640,
        overlap: float = 0.25,
        class_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 10000,
        apply_depth_filter: bool = True,
    ):
        """
        Full pipeline using tiled inference:
          - tiled detection -> merged boxes
          - bundle clustering + distances
        Returns: (annotated_rgb, count, error, bundle_info)

        NOTE: annotated_rgb returned is the original image (no seg mask merging).
        Your main.py can draw overlays on top.
        """
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            return None, 0, "Empty image passed to detect_rebars_tiled()", None

        boxes, scores, err = self._infer_boxes_tiled(
            image_bgr=image_bgr,
            model=model,
            depth_map=depth_map,
            tile_size=tile_size,
            overlap=overlap,
            class_id=class_id,
            conf=conf,
            iou=iou,
            max_det=max_det,
            apply_depth_filter=apply_depth_filter,
        )
        if err:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0, err, None

        if boxes is None or len(boxes) == 0:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0, None, None

        bundle_info = self.detect_bundles(boxes, depth_map)
        count = int(len(boxes))

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), count, None, bundle_info

    # ------------------------------------------------------------------
    # Service methods
    # ------------------------------------------------------------------
    def detect_image(self, file) -> Dict[str, Any]:
        if hasattr(file, "read"):
            content = file.read()
        elif isinstance(file, (bytes, bytearray)):
            content = file
        else:
            raise ValueError("Unsupported file type")

        nparr = np.frombuffer(content, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return {"count": 0, "error": "Failed to decode image.", "bundle_info": None, "image": None}

        annotated_rgb, count, error, bundle_info = self.detect_rebars(image_bgr, model, depth_map=None)

        # For uploaded images, annotate all bundles (numbers only; no bundle boxes here)
        if bundle_info and bundle_info.get("bundles"):
            counted_bundle_ids = {b["bundle_id"] for b in bundle_info.get("bundles", [])}
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            annotated_bgr = self.annotate_counted_bundles(annotated_bgr, None, bundle_info, counted_bundle_ids)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {"count": int(count), "error": error, "bundle_info": bundle_info, "image": image_data}

    def detect_oak_camera(self, frame_bgr: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
        annotated_rgb, count, error, bundle_info = self.detect_rebars(frame_bgr, model, depth_map=depth_map)
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None
        return {"count": int(count), "error": error, "bundle_info": bundle_info, "image": image_data}


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def _parse_in_hw(onnx_input_shape, default_hw=(640, 640)):
    def _to_int(v, default):
        try:
            iv = int(v)
            return iv if iv > 0 else default
        except Exception:
            return default

    if isinstance(onnx_input_shape, (list, tuple)) and len(onnx_input_shape) == 4:
        H = _to_int(onnx_input_shape[2], default_hw[0])
        W = _to_int(onnx_input_shape[3], default_hw[1])
        return (H, W)
    return default_hw


def _validate_model_path(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")
    if not model_path.lower().endswith(".onnx"):
        raise ValueError(f"Expected a .onnx file, got: {os.path.splitext(model_path)[1]}")


def _create_session(model_path: str):
    _validate_model_path(model_path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model at {model_path}. Error: {e}")

    in_name = sess.get_inputs()[0].name
    in_hw = _parse_in_hw(sess.get_inputs()[0].shape, default_hw=(640, 640))
    return {"sess": sess, "in_name": in_name, "in_hw": in_hw, "path": model_path}


def load_model():
    return _create_session(MODEL_PATH)


model = load_model()


# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------
def save_image_files(image_rgb: np.ndarray, det_id: str):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_path = os.path.join(DET_DIR, f"{det_id}.jpg")
    thumb_path = os.path.join(THUMB_DIR, f"{det_id}.jpg")

    cv2.imwrite(img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    h, w = img_bgr.shape[:2]
    tw = 360
    th = int(h * (tw / w)) if w else h
    thumb = cv2.resize(img_bgr, (tw, max(1, th)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(thumb_path, thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return img_path, thumb_path


def record_detection(
    user_id: int,
    processed_rgb: np.ndarray,
    count: int,
    stream_url: str,
    snapshot_url: str,
    bundle_info: dict = None,
):
    det_id = str(uuid.uuid4())
    img_path, thumb_path = save_image_files(processed_rgb, det_id)
    h, w = processed_rgb.shape[:2]

    conn = get_conn()
    cur = conn.cursor()
    bundle_json = None
    if bundle_info:
        import json

        safe_bundle_info = RebarBundleDetector()._make_json_safe(bundle_info)
        bundle_json = json.dumps(
            {
                "total_bundles": safe_bundle_info.get("total_bundles", 0),
                "rebars_in_bundles": safe_bundle_info.get("total_rebars_in_bundles", 0),
                "isolated": safe_bundle_info.get("total_isolated", 0),
                "nearest_bundle": safe_bundle_info.get("nearest_bundle"),
                "bundles": [
                    {
                        "bundle_id": b["bundle_id"],
                        "size": b["size"],
                        "distance_m": b.get("distance_m"),
                        "distance_mm": b.get("distance_mm"),
                        "is_nearest": b.get("is_nearest", False),
                        "counted": b.get("counted", True),
                    }
                    for b in safe_bundle_info.get("bundles", [])
                ],
            }
        )

    cur.execute(
        """
        INSERT INTO detections (
            id, user_id, timestamp, stream_url, snapshot_url,
            image_path, thumb_path, count, width, height, bundle_info
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            det_id,
            user_id,
            utc_now_iso(),
            stream_url,
            snapshot_url,
            img_path,
            thumb_path,
            int(count),
            int(w),
            int(h),
            bundle_json,
        ),
    )
    conn.commit()
    conn.close()
    return det_id


def list_detections(user_id: int, page: int, per_page: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM detections WHERE user_id=%s", (user_id,))
    total = cur.fetchone()["c"]

    offset = (page - 1) * per_page
    cur.execute(
        """
        SELECT * FROM detections
        WHERE user_id=%s
        ORDER BY timestamp DESC
        LIMIT %s OFFSET %s
        """,
        (user_id, per_page, offset),
    )
    rows = cur.fetchall()
    conn.close()
    return rows, total


def get_detection(det_id: str, user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM detections WHERE id=%s AND user_id=%s LIMIT 1", (det_id, user_id))
    row = cur.fetchone()
    conn.close()

    if row and row.get("bundle_info"):
        import json
        try:
            row["bundle_info"] = json.loads(row["bundle_info"])
        except Exception:
            pass
    return row


def delete_detection(det_id: str, user_id: int):
    row = get_detection(det_id, user_id)
    if not row:
        return False

    try:
        if os.path.exists(row["image_path"]):
            os.remove(row["image_path"])
        if os.path.exists(row["thumb_path"]):
            os.remove(row["thumb_path"])
    except Exception:
        pass

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM detections WHERE id=%s AND user_id=%s", (det_id, user_id))
    conn.commit()
    conn.close()
    return True


# ------------------------------------------------------------------
# Image to data URI helpers
# ------------------------------------------------------------------
def img_to_data_uri(
    image_rgb: np.ndarray, quality: int = 90, max_w: Optional[int] = None
) -> Optional[str]:
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
        return None

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if max_w is not None:
        h, w = bgr.shape[:2]
        if w > max_w:
            new_h = int(h * (max_w / w))
            bgr = cv2.resize(bgr, (max_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def file_to_data_uri(path: str, max_w: int = 120, quality: int = 85) -> Optional[str]:
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, max(1, int(h * (max_w / w)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")