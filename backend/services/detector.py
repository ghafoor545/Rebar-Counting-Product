# backend/services/detector.py

import os
import uuid
import base64
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

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
        eps: float = 1.0,
        min_bundle_size: int = 3,
        min_samples: int = 2,
        row_tolerance: float = 40.0,
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
        self.eps = eps
        self.min_bundle_size = min_bundle_size
        self.min_samples = min_samples
        self.row_tolerance = row_tolerance
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

    def _make_json_safe(self, obj):
        """Recursively convert non-JSON-serializable values to None"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_safe(obj.tolist())
        elif isinstance(obj, (float, int)):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
        return obj

    def _calculate_adaptive_eps(self, centers: np.ndarray, sizes: np.ndarray) -> float:
        """Calculate adaptive eps based on rebar sizes and distances"""
        if len(centers) < 2:
            return self.eps
        
        avg_size = float(np.mean(sizes))
        
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                distances.append(dist)
        
        if distances:
            median_distance = np.median(distances)
            dynamic_eps = max(avg_size * 0.5, median_distance * 0.3)
            dynamic_eps = min(max(dynamic_eps, self.eps * 0.5), self.eps * 2.0)
            return float(dynamic_eps)
        
        return float(self.eps)

    def _calculate_bundle_distance(
        self, 
        bundle: Dict[str, Any], 
        depth_map: np.ndarray,
        boxes: np.ndarray
    ) -> Optional[float]:
        """Calculate median distance for a bundle based on its member rebar distances"""
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
                print(f"[Bundle {bundle['bundle_id']}] Distance: {bundle_distance:.0f}mm "
                      f"({bundle_distance/1000:.2f}m) from {len(distances)} rebars")
            
            return bundle_distance
        else:
            bundle["distance_mm"] = None
            bundle["distance_m"] = None
            return None

    # ------------------------------------------------------------------
    # Sort rebars from top to bottom
    # ------------------------------------------------------------------
    def _sort_rebars_top_to_bottom_by_indices(self, indices, centers):
        """Sort rebars from top to bottom by y-coordinate (smaller y = top)"""
        if len(indices) <= 1:
            return indices
        
        pairs = list(zip(indices, centers))
        # Sort by y-coordinate (top to bottom)
        pairs.sort(key=lambda x: x[1][1])
        
        return np.array([idx for idx, _ in pairs])

    # ------------------------------------------------------------------
    # Bundle clustering
    # ------------------------------------------------------------------
    def detect_bundles(
        self, 
        boxes: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        if len(boxes) == 0:
            return self._make_json_safe({
                "bundles": [],
                "isolated_rebars": [],
                "total_bundles": 0,
                "total_rebars_in_bundles": 0,
                "total_isolated": 0,
                "total_count": 0,
                "id_mapping": {},
                "display_summary": {
                    "total": 0,
                    "bundles": 0,
                    "rebars_in_bundles": 0,
                    "isolated": 0,
                },
                "nearest_bundle": None,
            })

        centers, sizes = self._get_box_centers_and_sizes(boxes)

        if self.use_adaptive_eps:
            eps_to_use = self._calculate_adaptive_eps(centers, sizes)
            if self.debug:
                print(f"[DBSCAN] Dynamic eps: {eps_to_use:.2f}")
        else:
            eps_to_use = float(self.eps)

        clustering = DBSCAN(eps=eps_to_use, min_samples=int(self.min_samples)).fit(centers)
        labels = clustering.labels_

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        bundle_clusters = [
            label
            for label in set(labels.tolist())
            if label != -1 and cluster_sizes.get(label, 0) >= int(self.min_bundle_size)
        ]

        bundles: List[Dict[str, Any]] = []
        isolated_rebars: List[Dict[str, Any]] = []
        id_mapping: Dict[int, Dict[str, Any]] = {}
        bundle_counter = 0

        unassigned = set(range(len(boxes)))

        min_neighbors = (
            (int(self.min_samples) - 1)
            if self.min_neighbors_in_bundle is None
            else int(self.min_neighbors_in_bundle)
        )
        min_neighbors = max(1, min_neighbors)

        for label in bundle_clusters:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_centers = centers[cluster_indices]

            diff = cluster_centers[:, None, :] - cluster_centers[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            neighbor_counts = (dist <= eps_to_use).sum(axis=1) - 1

            keep_local = neighbor_counts >= min_neighbors
            kept_indices = cluster_indices[keep_local]

            if len(kept_indices) < int(self.min_bundle_size):
                continue

            bundle_counter += 1
            kept_centers = centers[kept_indices]
            kept_boxes = boxes[kept_indices]
            cluster_size = int(len(kept_indices))

            # Sort rebars from top to bottom
            sorted_indices = self._sort_rebars_top_to_bottom_by_indices(
                kept_indices, kept_centers
            )

            bundle_rebars = []
            # Assign IDs starting from 1 for each bundle (top to bottom)
            for bundle_idx, global_idx in enumerate(sorted_indices, start=1):
                bundle_rebars.append(
                    {
                        "global_index": int(global_idx),
                        "bundle_index": int(bundle_idx),
                        "display_id": int(bundle_idx),  # Bundle-specific ID starting at 1
                        "box": boxes[global_idx].tolist(),
                        "center": centers[global_idx].tolist(),
                        "row": self._get_row_number(centers[global_idx], kept_centers),
                    }
                )
                id_mapping[int(global_idx)] = {
                    "display_id": int(bundle_idx),  # This is the ID that gets drawn
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

        # Handle nearest bundle only mode
        nearest_bundle_info = None
        
        if self.nearest_bundle_only and bundles:
            valid_bundles = [b for b in bundles if b.get("distance_mm") is not None]
            if valid_bundles:
                nearest_bundle = min(valid_bundles, key=lambda b: b["distance_mm"])
                
                for bundle in bundles:
                    bundle["is_nearest"] = (bundle["bundle_id"] == nearest_bundle["bundle_id"])
                    bundle["counted"] = (bundle["bundle_id"] == nearest_bundle["bundle_id"])
                
                if self.debug:
                    print(f"[Nearest Bundle Mode] Keeping only bundle {nearest_bundle['bundle_id']}")
                
                bundles = [nearest_bundle]
                
                nearest_rebar_indices = set(nearest_bundle["global_indices"])
                
                new_id_mapping = {}
                for idx in nearest_rebar_indices:
                    if idx in id_mapping:
                        new_id_mapping[idx] = id_mapping[idx]
                id_mapping = new_id_mapping
                
                isolated_indices = [i for i in unassigned if i not in nearest_rebar_indices]
                isolated_rebars = []
                for display_idx, global_idx in enumerate(isolated_indices, start=1):
                    isolated_rebars.append({
                        "global_index": int(global_idx),
                        "display_id": int(display_idx),
                        "box": boxes[global_idx].tolist(),
                        "center": centers[global_idx].tolist(),
                        "type": "isolated",
                        "group_size": 1,
                    })
                
                total_rebars_in_bundles = nearest_bundle["size"]
                total_isolated = len(isolated_indices)
                total_bundles = 1
                
                nearest_bundle_info = {
                    "bundle_id": nearest_bundle["bundle_id"],
                    "distance_mm": nearest_bundle["distance_mm"],
                    "distance_m": nearest_bundle["distance_m"],
                    "size": nearest_bundle["size"]
                }
                
                display_summary = {
                    "total": int(len(boxes)),
                    "bundles": total_bundles,
                    "rebars_in_bundles": total_rebars_in_bundles,
                    "isolated": total_isolated,
                }
                
                result = {
                    "bundles": bundles,
                    "isolated_rebars": isolated_rebars,
                    "total_bundles": total_bundles,
                    "total_rebars_in_bundles": total_rebars_in_bundles,
                    "total_isolated": total_isolated,
                    "total_count": int(len(boxes)),
                    "id_mapping": id_mapping,
                    "display_summary": display_summary,
                    "nearest_bundle": nearest_bundle_info,
                }
                return self._make_json_safe(result)
        
        # Normal mode - mark all as counted
        for bundle in bundles:
            bundle["counted"] = True
        
        total_bundles = int(len(bundles))
        
        if bundles and self.track_bundle_distances:
            valid_bundles = [b for b in bundles if b.get("distance_mm") is not None]
            if valid_bundles:
                nearest = min(valid_bundles, key=lambda b: b["distance_mm"])
                nearest_bundle_info = {
                    "bundle_id": nearest["bundle_id"],
                    "distance_mm": nearest["distance_mm"],
                    "distance_m": nearest["distance_m"],
                    "size": nearest["size"]
                }
                for bundle in bundles:
                    bundle["is_nearest"] = (bundle["bundle_id"] == nearest["bundle_id"])
        
        result = {
            "bundles": bundles,
            "isolated_rebars": isolated_rebars,
            "total_bundles": total_bundles,
            "total_rebars_in_bundles": 0,
            "total_isolated": int(len(isolated_rebars)),
            "total_count": int(len(boxes)),
            "id_mapping": id_mapping,
            "display_summary": {
                "total": int(len(boxes)),
                "bundles": total_bundles,
                "rebars_in_bundles": 0,
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
    # Depth-based filtering
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
            
            depth_mm = get_depth_at_point(depth_map, cx, cy, sample_radius=5)
            
            if depth_mm is not None:
                if (self.min_detection_distance_mm <= depth_mm <= self.max_detection_distance_mm):
                    keep_mask[i] = True
        
        filtered_boxes = boxes[keep_mask]
        filtered_scores = scores[keep_mask]
        filtered_coeffs = coeffs[keep_mask] if coeffs is not None else None
        
        return filtered_boxes, filtered_scores, filtered_coeffs

    # ------------------------------------------------------------------
    # Drawing methods - Annotate only counted rebars with bundle-specific IDs
    # ------------------------------------------------------------------
    def annotate_counted_bundles(self, image_bgr, boxes, bundle_info, counted_bundle_ids):
        """
        Annotate ONLY the rebars in bundles that are counted.
        Each bundle has its own IDs starting from 1 (top to bottom).
        
        Args:
            image_bgr: Original image
            boxes: All detection boxes
            bundle_info: Bundle info from detect_bundles
            counted_bundle_ids: Set of bundle IDs that should be annotated
        """
        img = image_bgr.copy()
        
        if not bundle_info:
            return img
        
        bundles = bundle_info.get("bundles", [])
        id_mapping = bundle_info.get("id_mapping", {})
        
# Annotate counted rebars with bundle-specific 1..N IDs for each bundle (_within bundle_ number reset),
        # sorted top-to-bottom (by center y), then left-to-right (by center x).
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

                # Use smaller font for small rebars to reduce clutter, always centered.
                font_scale = max(0.16, min(0.4, box_size / 140.0))
                thickness = 1

                (text_w, text_h), _ = cv2.getTextSize(display_txt, font, font_scale, thickness)

                # If the text is too big for the rebar, scale it down.
                if text_w > box_size * 0.8 or text_h > box_size * 0.8:
                    scale_factor = min(box_size * 0.8 / text_w, box_size * 0.8 / text_h)
                    font_scale *= scale_factor
                    font_scale = max(0.14, font_scale)
                    (text_w, text_h), _ = cv2.getTextSize(display_txt, font, font_scale, thickness)

                text_x = cx - text_w // 2
                text_y = cy + text_h // 2

                padding = max(2, int(text_h * 0.2))
                rect_x1 = text_x - padding
                rect_y1 = text_y - text_h - padding
                rect_x2 = text_x + text_w + padding
                rect_y2 = text_y + padding

                # Keep label inside image boundaries
                h, w = img.shape[:2]
                rect_x1 = max(0, min(rect_x1, w - 1))
                rect_x2 = max(0, min(rect_x2, w - 1))
                rect_y1 = max(0, min(rect_y1, h - 1))
                rect_y2 = max(0, min(rect_y2, h - 1))

                # Draw a soft shadowed text only (no heavy rectangle/marker) for readability
                cv2.putText(
                    img,
                    display_txt,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    img,
                    display_txt,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return img

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
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return im, r, (left, top)

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

        if len(arrs) == 2 and arrs[0].ndim == 3 and arrs[1].ndim == 4:
            return "seg", (arrs[0], arrs[1])

        for a in arrs:
            if a.ndim == 3 and a.shape[-1] == 6:
                return "nms", (a[0] if a.shape[0] == 1 else a)
            if a.ndim == 2 and a.shape[-1] == 6:
                return "nms", a

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
    # Core detect
    # ------------------------------------------------------------------
    def detect_rebars(
        self,
        image_bgr,
        model,
        depth_map: Optional[np.ndarray] = None,
        class_id: int = 0,
        conf: float = 0.5,
        iou: float = 0.3,
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
                    (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0]) &
                    (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
                ).sum()
                b_xyxy_as_is = boxes.copy()
                valid2 = (
                    (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0]) &
                    (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
                ).sum()
                b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

                keep_mask = scores >= float(conf)
                if np.any(keep_mask):
                    b_xyxy = b_xyxy[keep_mask]
                    s = scores[keep_mask]
                    coeffs = coeffs[keep_mask]

                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    wh = (b_xyxy[:, 2:4] - b_xyxy[:, 0:2]).clip(min=0)
                    small = (wh[:, 0] < 4.0) | (wh[:, 1] < 4.0)
                    if np.any(small):
                        b_xyxy = b_xyxy[~small]
                        s = s[~small]
                        coeffs = coeffs[~small]

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
                            b_xyxy, s, coeffs = self.filter_by_depth(
                                b_xyxy, s, coeffs, depth_map
                            )

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
                                instance_colors = [self._color_for_index(i) for i in range(len(seg_masks))]

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
                    (b_xyxy_from_xywh[:, 2] > b_xyxy_from_xywh[:, 0]) &
                    (b_xyxy_from_xywh[:, 3] > b_xyxy_from_xywh[:, 1])
                ).sum()
                b_xyxy_as_is = boxes.copy()
                valid2 = (
                    (b_xyxy_as_is[:, 2] > b_xyxy_as_is[:, 0]) &
                    (b_xyxy_as_is[:, 3] > b_xyxy_as_is[:, 1])
                ).sum()
                b_xyxy = b_xyxy_from_xywh if valid1 >= valid2 else b_xyxy_as_is

                keep_mask = scores >= float(conf)
                if np.any(keep_mask):
                    b_xyxy = b_xyxy[keep_mask]
                    s = scores[keep_mask]

                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    wh = (b_xyxy[:, 2:4] - b_xyxy[:, 0:2]).clip(min=0)
                    small = (wh[:, 0] < 4.0) | (wh[:, 1] < 4.0)
                    if np.any(small):
                        b_xyxy = b_xyxy[~small]
                        s = s[~small]

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

            # Return clean image without annotations (annotations will be added in main.py)
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), count, None, bundle_info

        except Exception as e:
            try:
                fallback = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                fallback = None
            return fallback, 0, f"Detection error: {e}", None

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
            return {
                "count": 0,
                "error": "Failed to decode image.",
                "bundle_info": None,
                "image": None,
            }

        annotated_rgb, count, error, bundle_info = self.detect_rebars(image_bgr, model, depth_map=None)
        
        # For uploaded images, annotate all bundles
        if bundle_info and bundle_info.get("bundles"):
            all_boxes = []
            for bundle in bundle_info.get("bundles", []):
                for rebar in bundle.get("rebars", []):
                    all_boxes.append(rebar["box"])
            if all_boxes:
                boxes_array = np.array(all_boxes, dtype=np.float32)
                counted_bundle_ids = set([b["bundle_id"] for b in bundle_info.get("bundles", [])])
                annotated_result = self.annotate_counted_bundles(
                    image_bgr, boxes_array, bundle_info, counted_bundle_ids
                )
                annotated_rgb = cv2.cvtColor(annotated_result, cv2.COLOR_BGR2RGB)
        
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {
            "count": int(count),
            "error": error,
            "bundle_info": bundle_info,
            "image": image_data,
        }

    def detect_oak_camera(
        self, 
        frame_bgr: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        annotated_rgb, count, error, bundle_info = self.detect_rebars(
            frame_bgr, 
            model,
            depth_map=depth_map
        )
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {
            "count": int(count),
            "error": error,
            "bundle_info": bundle_info,
            "image": image_data,
        }


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
        bundle_json = json.dumps({
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
                    "counted": b.get("counted", True)
                }
                for b in safe_bundle_info.get("bundles", [])
            ]
        })

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
    cur.execute(
        "SELECT * FROM detections WHERE id=%s AND user_id=%s LIMIT 1",
        (det_id, user_id),
    )
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
        img = cv2.resize(img, (max_w, int(h * (max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
