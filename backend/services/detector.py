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
    """
    Detects bundles of rebars based on proximity clustering.

    Bundle definition: min_bundle_size or more rebars close to each other.
    Less than min_bundle_size rebars together are considered isolated.

    FIX (bundle counting):
      DBSCAN can "absorb" border points into a cluster. We prune weakly connected
      members inside each candidate bundle by requiring min neighbor count.

    Supports segmentation ONNX:
      output0: [B, no, anchors]  (e.g. [1, 37, 8400])
      output1: [B, nm, mh, mw]   (e.g. [1, 32, 160, 160])

    Visualization:
      - Different color mask for each instance
      - Bounding box colored to match mask
      - IMPORTANT: masks are HARD-CLIPPED to each bbox in original image space
        to remove rainbow noise in background.
    """

    def __init__(
        self,
        eps: float = 1.0,
        min_bundle_size: int = 5,
        min_samples: int = 4,
        row_tolerance: float = 10.0,
        use_adaptive_eps: bool = True,
        min_neighbors_in_bundle: Optional[int] = None,
        # segmentation overlay
        draw_seg_masks: bool = True,
        seg_mask_thresh: float = 0.55,
        seg_mask_alpha: float = 0.35,
        draw_mask_contours: bool = True,
        mask_contour_thickness: int = 2,
        # debug
        debug: bool = False,
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

        self.debug = debug

    # ------------------------------------------------------------------
    # Bundle clustering
    # ------------------------------------------------------------------
    def detect_bundles(self, boxes: np.ndarray) -> Dict[str, Any]:
        if len(boxes) == 0:
            return {
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
            }

        centers, sizes = self._get_box_centers_and_sizes(boxes)

        eps_to_use = float(self.eps)
        if self.use_adaptive_eps and len(sizes):
            avg_size = float(np.mean(sizes))
            eps_to_use = max(float(self.eps), avg_size * 0.10)

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

        # Everything starts unassigned; only accepted bundle members are removed.
        unassigned = set(range(len(boxes)))

        min_neighbors = (
            (int(self.min_samples) - 1)
            if self.min_neighbors_in_bundle is None
            else int(self.min_neighbors_in_bundle)
        )
        min_neighbors = max(1, min_neighbors)

        # ---- Process bundles with pruning ----
        for label in bundle_clusters:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_centers = centers[cluster_indices]

            # Neighbor counts inside cluster (within eps_to_use)
            diff = cluster_centers[:, None, :] - cluster_centers[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
            neighbor_counts = (dist <= eps_to_use).sum(axis=1) - 1  # exclude self

            keep_local = neighbor_counts >= min_neighbors
            kept_indices = cluster_indices[keep_local]

            # If pruning makes it too small, not a bundle
            if len(kept_indices) < int(self.min_bundle_size):
                continue

            bundle_counter += 1
            kept_centers = centers[kept_indices]
            kept_boxes = boxes[kept_indices]
            cluster_size = int(len(kept_indices))

            sorted_indices = self._sort_rebars_rowwise_left_to_right(
                kept_indices, kept_centers
            )

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

            bundles.append(
                {
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
                }
            )

            for gi in kept_indices.tolist():
                unassigned.discard(int(gi))

        # ---- Isolated = everything not accepted into a bundle ----
        isolated_indices = sorted(list(unassigned))
        if isolated_indices:
            isolated_centers = centers[isolated_indices]
            sorted_isolated = self._sort_rebars_rowwise_left_to_right(
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

        total_rebars_in_bundles = int(sum(b["size"] for b in bundles))
        total_isolated = int(len(isolated_rebars))

        return {
            "bundles": bundles,
            "isolated_rebars": isolated_rebars,
            "total_bundles": int(len(bundles)),
            "total_rebars_in_bundles": total_rebars_in_bundles,
            "total_isolated": total_isolated,
            "total_count": int(len(boxes)),
            "id_mapping": id_mapping,
            "display_summary": {
                "total": int(len(boxes)),
                "bundles": int(len(bundles)),
                "rebars_in_bundles": total_rebars_in_bundles,
                "isolated": total_isolated,
            },
        }

    def _sort_rebars_rowwise_left_to_right(self, indices, centers):
        if len(indices) <= 1:
            return indices

        pairs = list(zip(indices, centers))
        pairs.sort(key=lambda x: x[1][1])  # by y

        rows = []
        current_row = []
        current_y = pairs[0][1][1]

        for idx, center in pairs:
            if abs(center[1] - current_y) <= self.row_tolerance:
                current_row.append((idx, center))
            else:
                if current_row:
                    current_row.sort(key=lambda x: x[1][0])  # by x
                    rows.extend([i for i, _ in current_row])
                current_row = [(idx, center)]
                current_y = center[1]

        if current_row:
            current_row.sort(key=lambda x: x[1][0])
            rows.extend([i for i, _ in current_row])

        return np.array(rows)

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
    # Detection helpers & ONNX inference
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
        shape = im.shape[:2]  # (h, w)
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

        # ---- YOLO Segmentation ONNX: (pred, proto) ----
        if len(arrs) == 2 and arrs[0].ndim == 3 and arrs[1].ndim == 4:
            return "seg", (arrs[0], arrs[1])

        # ---- NMS-like outputs ----
        for a in arrs:
            if a.ndim == 3 and a.shape[-1] == 6:
                return "nms", (a[0] if a.shape[0] == 1 else a)
            if a.ndim == 2 and a.shape[-1] == 6:
                return "nms", a

        # ---- fallback raw outputs ----
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
    # Segmentation masks (decoded + HARD-CLIPPED to bbox)
    # ------------------------------------------------------------------
    def decode_seg_masks(
        self,
        coeffs: np.ndarray,          # [N, nm]
        proto: np.ndarray,           # [nm, mh, mw]
        boxes_xyxy_orig: np.ndarray, # [N, 4] in ORIGINAL image coords (after scale_boxes + NMS)
        r: float,
        dwdh: Tuple[float, float],
        orig_shape,
        in_hw: Tuple[int, int],
        mask_thresh: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Decode YOLO-seg masks and HARD-CLIP each mask to its own bounding box in original image space.
        This removes the background rainbow noise completely.
        """
        if coeffs is None or len(coeffs) == 0:
            return []

        in_h, in_w = in_hw
        orig_h, orig_w = orig_shape[:2]
        left, top = map(int, dwdh)

        proto = proto.astype(np.float32)      # [nm, mh, mw]
        coeffs = coeffs.astype(np.float32)    # [N, nm]

        nm, mh, mw = proto.shape
        proto_flat = proto.reshape(nm, -1)    # [nm, mh*mw]

        # masks at proto resolution: [N, mh, mw]
        masks = self.sigmoid(coeffs @ proto_flat).reshape(-1, mh, mw)

        # content size in model-input after letterbox
        new_w = int(round(orig_w * float(r)))
        new_h = int(round(orig_h * float(r)))

        out_masks: List[np.ndarray] = []
        for i, m in enumerate(masks):
            # 1) upsample proto mask -> model input size (e.g., 640x640)
            m_in = cv2.resize(m, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

            # 2) remove padding -> get mask on resized-content area
            x1p = max(left, 0)
            y1p = max(top, 0)
            x2p = min(left + new_w, in_w)
            y2p = min(top + new_h, in_h)
            if x2p <= x1p or y2p <= y1p:
                out_masks.append(np.zeros((orig_h, orig_w), dtype=bool))
                continue

            m_crop = m_in[y1p:y2p, x1p:x2p]

            # 3) resize to original image size
            m_orig = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            m_bin = (m_orig >= float(mask_thresh))

            # 4) HARD CLIP mask to its bbox in ORIGINAL image coords (key fix)
            bx1, by1, bx2, by2 = boxes_xyxy_orig[i]
            x1 = int(max(0, np.floor(bx1)))
            y1 = int(max(0, np.floor(by1)))
            x2 = int(min(orig_w, np.ceil(bx2)))
            y2 = int(min(orig_h, np.ceil(by2)))

            # zero everything outside the box
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
        """
        Deterministic vivid BGR color for instance i (OpenCV uses BGR).
        """
        h = int((i * 37) % 180)  # 0..179 in OpenCV HSV
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

            # alpha blend only on mask pixels
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
    # Drawing
    # ------------------------------------------------------------------
    def draw_simple_black_frame(self, img, summary):
        frame_width = 220
        frame_height = 140
        x1, y1 = 10, 10
        x2, y2 = x1 + frame_width, y1 + frame_height

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        title = "REBAR ANALYSIS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, title, (x1 + 15, y1 + 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (x1 + 10, y1 + 35), (x2 - 10, y1 + 35), (255, 255, 255), 1)

        y_offset = y1 + 50
        line_height = 18
        cv2.putText(img, f"TOTAL: {summary['total']}", (x1 + 15, y_offset), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += line_height
        cv2.putText(img, f"BUNDLES: {summary['bundles']}", (x1 + 15, y_offset), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += line_height
        cv2.putText(img, f"IN BUNDLES: {summary['rebars_in_bundles']}", (x1 + 15, y_offset), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += line_height
        cv2.putText(img, f"ISOLATED: {summary['isolated']}", (x1 + 15, y_offset), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def draw_centered_ids_with_bundles(self, image_bgr, boxes, bundle_info=None, box_colors=None):
        """
        Draw per-instance colored bounding boxes (if box_colors provided),
        centered IDs, and summary frame.
        """
        img = image_bgr.copy()
        img_height, img_width = img.shape[:2]

        # Draw all boxes (instance-colored)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            if box_colors is not None and i < len(box_colors) and box_colors[i] is not None:
                color = tuple(map(int, box_colors[i]))
            else:
                color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw IDs using bundle mapping
        if bundle_info and bundle_info.get("total_count", 0) > 0:
            for global_idx, id_info in bundle_info["id_mapping"].items():
                if global_idx < len(boxes):
                    box = boxes[global_idx]
                    x1, y1, x2, y2 = map(int, box[:4])
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_size = min(box_w, box_h)

                    if box_size < 25:
                        font_scale, thickness = 0.25, 1
                    elif box_size < 40:
                        font_scale, thickness = 0.30, 1
                    elif box_size < 70:
                        font_scale, thickness = 0.35, 1
                    elif box_size < 120:
                        font_scale, thickness = 0.40, 2
                    elif box_size < 200:
                        font_scale, thickness = 0.50, 2
                    else:
                        font_scale, thickness = 0.60, 2

                    display_id = str(id_info["display_id"])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(display_id, font, font_scale, thickness)
                    padding = max(2, int(font_scale * 3))

                    bg_x1 = max(0, cx - tw // 2 - padding)
                    bg_y1 = max(0, cy - th // 2 - padding)
                    bg_x2 = min(img_width, cx + tw // 2 + padding)
                    bg_y2 = min(img_height, cy + th // 2 + padding)

                    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.putText(
                        img,
                        display_id,
                        (cx - tw // 2, cy + th // 2),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

            img = self.draw_simple_black_frame(img, bundle_info["display_summary"])
        else:
            # simple numbering fallback
            for idx, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = map(int, box[:4])
                box_size = min(x2 - x1, y2 - y1)

                if box_size < 30:
                    font_scale, thickness = 0.3, 1
                elif box_size < 60:
                    font_scale, thickness = 0.4, 1
                else:
                    font_scale, thickness = 0.5, 2

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                id_text = str(idx)
                (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                cv2.rectangle(
                    img,
                    (cx - tw // 2 - 2, cy - th // 2 - 2),
                    (cx + tw // 2 + 2, cy + th // 2 + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    img,
                    id_text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            summary = {
                "total": len(boxes),
                "bundles": 0,
                "rebars_in_bundles": 0,
                "isolated": len(boxes),
            }
            img = self.draw_simple_black_frame(img, summary)

        return img, boxes

    # ------------------------------------------------------------------
    # Core detect (supports DET + SEG ONNX)
    # ------------------------------------------------------------------
    def detect_rebars(
        self,
        image_bgr,
        model,
        class_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 10000,
    ):
        # Guard
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

            if self.debug:
                print("[detect_rebars] kind=", kind, "outs shapes=", [np.asarray(o).shape for o in outs])

            dets_xyxy = []
            seg_masks: Optional[List[np.ndarray]] = None
            instance_colors: Optional[List[Tuple[int, int, int]]] = None

            # ------------------------
            # SEGMENTATION branch
            # ------------------------
            if kind == "seg":
                pred_raw, proto_raw = data
                pred = pred_raw[0]   # [no, anchors] OR [anchors, no]
                proto = proto_raw[0] # [nm, mh, mw]

                # Your model output0 is [1, 37, anchors] => pred is [37, anchors] -> transpose
                if pred.ndim != 2:
                    raise ValueError(f"Unexpected seg pred ndim={pred.ndim}, shape={pred.shape}")
                if pred.shape[0] <= 128 and pred.shape[1] > pred.shape[0]:
                    pred = pred.T  # [anchors, 37]

                pred = pred.astype(np.float32)
                no = int(pred.shape[1])
                nm = int(proto.shape[0])

                # YOLO-seg typical: no = 4 + nc + nm  (your: 37=4+1+32)
                nc = no - 4 - nm
                if nc <= 0:
                    nc = 1

                boxes = pred[:, 0:4].copy()

                # scores (class probability)
                scores_mat = self.maybe_sigmoid(pred[:, 4 : 4 + nc])
                if scores_mat.ndim == 1:
                    scores_mat = scores_mat[:, None]
                if int(class_id) >= scores_mat.shape[1]:
                    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0, "class_id out of range", None
                scores = scores_mat[:, int(class_id)]

                # mask coeffs
                coeff_start = 4 + nc
                coeff_end = coeff_start + nm
                if coeff_end > no:
                    # fallback if export uses [x,y,w,h,conf,coeffs...]
                    coeff_start = 5
                    coeff_end = 5 + nm
                coeffs = pred[:, coeff_start:coeff_end].copy()
                if coeffs.shape[1] != nm:
                    raise ValueError(f"Bad coeffs shape {coeffs.shape}, expected nm={nm}, no={no}")

                # normalized -> pixels in model input
                if np.nanmax(boxes) <= 1.5:
                    boxes[:, [0, 2]] *= float(in_w)
                    boxes[:, [1, 3]] *= float(in_h)

                # decide xywh vs xyxy
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

                    # scale to original image coords
                    b_xyxy = self.scale_boxes(b_xyxy, r, dwdh, image_bgr.shape)

                    # remove tiny
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
                        coeffs = coeffs[keep]
                        dets_xyxy = b_xyxy.tolist()

                        if self.draw_seg_masks:
                            # HARD-CLIP masks to bbox -> removes background rainbow
                            seg_masks = self.decode_seg_masks(
                                coeffs=coeffs,
                                proto=proto,
                                boxes_xyxy_orig=b_xyxy,  # IMPORTANT
                                r=float(r),
                                dwdh=dwdh,
                                orig_shape=image_bgr.shape,
                                in_hw=in_hw,
                                mask_thresh=float(self.seg_mask_thresh),
                            )
                            instance_colors = [self._color_for_index(i) for i in range(len(seg_masks))]

            # ------------------------
            # DETECTION (NMS-like) branch
            # ------------------------
            elif kind == "nms":
                d = data.reshape(-1, 6).astype(np.float32)
                cls = np.round(d[:, 5]).astype(np.int32)
                scr = d[:, 4]
                mask = (scr >= conf) & (cls == int(class_id))
                if np.any(mask):
                    b = d[mask, :4]
                    if np.nanmax(b) <= 1.5:
                        b[:, [0, 2]] *= float(in_w)
                        b[:, [1, 3]] *= float(in_h)
                    b = self.scale_boxes(b, r, dwdh, image_bgr.shape)
                    dets_xyxy = b.tolist()

            # ------------------------
            # DETECTION (raw YOLO-style) branch
            # ------------------------
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
                        dets_xyxy = b_xyxy.tolist()

            # ------------------------
            # Build annotation + bundle info
            # ------------------------
            if dets_xyxy:
                boxes_array = np.array(dets_xyxy, dtype=np.float32)

                annotated = image_bgr.copy()

                # Overlay instance-colored masks (already hard-clipped to bbox)
                if seg_masks and instance_colors:
                    annotated = self._overlay_instance_masks(
                        annotated,
                        seg_masks,
                        instance_colors,
                        alpha=float(self.seg_mask_alpha),
                        draw_contours=bool(self.draw_mask_contours),
                        contour_thickness=int(self.mask_contour_thickness),
                    )

                bundle_info = self.detect_bundles(boxes_array)

                # Box colors match instance mask colors
                box_colors = instance_colors if instance_colors and len(instance_colors) == len(boxes_array) else None

                annotated, sorted_boxes = self.draw_centered_ids_with_bundles(
                    annotated, boxes_array, bundle_info, box_colors=box_colors
                )
                count = int(len(sorted_boxes))
            else:
                annotated = image_bgr.copy()
                bundle_info = None
                count = 0

            # HD output with banner
            hd = self.to_hd_1080p(annotated, background=(18, 24, 31))
            banner_height = 100
            img_h, img_w, _ = hd.shape

            if bundle_info and bundle_info.get("total_bundles", 0) > 0:
                heading = (
                    f"Rebars: {count} | Bundles: {bundle_info['total_bundles']} | "
                    f"In Bundles: {bundle_info['total_rebars_in_bundles']} | "
                    f"Isolated: {bundle_info['total_isolated']}"
                )
            else:
                heading = f"Rebars detected: {count}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            size, _ = cv2.getTextSize(heading, font, 1.5, 3)
            cv2.rectangle(hd, (0, 0), (img_w, banner_height), (255, 255, 255), -1)
            cv2.putText(
                hd,
                heading,
                ((img_w - size[0]) // 2, (banner_height + size[1]) // 2),
                font,
                1.5,
                (0, 0, 0),
                3,
                lineType=cv2.LINE_AA,
            )

            return cv2.cvtColor(hd, cv2.COLOR_BGR2RGB), count, None, bundle_info

        except Exception as e:
            try:
                fallback = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                fallback = None
            return fallback, 0, f"Detection error: {e}", None

    # ------------------------------------------------------------------
    # FastAPI-oriented service methods (JSON responses)
    # ------------------------------------------------------------------
    def detect_image(self, file) -> Dict[str, Any]:
        if hasattr(file, "read"):
            content = file.read()
        elif isinstance(file, (bytes, bytearray)):
            content = file
        else:
            raise ValueError("Unsupported file type for detect_image; expected bytes or file-like object.")

        nparr = np.frombuffer(content, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return {
                "count": 0,
                "error": "Failed to decode image.",
                "bundle_info": None,
                "image": None,
            }

        annotated_rgb, count, error, bundle_info = self.detect_rebars(image_bgr, model)
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {
            "count": int(count),
            "error": error,
            "bundle_info": bundle_info,
            "image": image_data,
        }

    def detect_oak_camera(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        annotated_rgb, count, error, bundle_info = self.detect_rebars(frame_bgr, model)
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {
            "count": int(count),
            "error": error,
            "bundle_info": bundle_info,
            "image": image_data,
        }


# ------------------------------------------------------------------
# Model loading (ONNX Runtime)
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
# Detection record helpers (DB)
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

        bundle_json = json.dumps(
            {
                "total_bundles": bundle_info.get("total_bundles", 0),
                "rebars_in_bundles": bundle_info.get("total_rebars_in_bundles", 0),
                "isolated": bundle_info.get("total_isolated", 0),
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
# Image → data URI helpers
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
