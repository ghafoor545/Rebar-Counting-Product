import os
import uuid
import base64
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2
import requests
import onnxruntime as ort
from sklearn.cluster import DBSCAN
from backend.core.config import MODEL_PATH, DET_DIR, THUMB_DIR
from backend.db import get_conn
from backend.utils.utils import utc_now_iso

# Bundle Detector Class
class RebarBundleDetector:
    """
    Detects bundles of rebars based on proximity clustering
    Bundle definition: 5 or more rebars close to each other
    Less than 5 rebars together are considered isolated

    ID Assignment Rules:
        Each bundle has its own separate numbering (1,2,3... within the bundle)
        Within each bundle: IDs assigned row-wise, left to right
        Isolated rebars: sequential numbers (continuing from bundles)
    """

    def __init__(
        self,
        eps: float = 100.0,  # Distance threshold for clustering (pixels)
        min_bundle_size: int = 5,  # Minimum rebars to form a bundle
        min_samples: int = 2,  # Min samples for DBSCAN
        row_tolerance: float = 40.0,  # Pixel tolerance for same row
        use_adaptive_eps: bool = True,  # Use adaptive eps based on rebar sizes
    ):
        self.eps = eps
        self.min_bundle_size = min_bundle_size
        self.min_samples = min_samples
        self.row_tolerance = row_tolerance
        self.use_adaptive_eps = use_adaptive_eps

    # ------------------------------------------------------------------
    # Bundle clustering
    # ------------------------------------------------------------------
    def detect_bundles(self, boxes: np.ndarray) -> Dict[str, Any]:
        """
        Detect bundles from detected rebar boxes and assign bundle-specific IDs.
        IDs are assigned left-to-right in rows, top-to-bottom WITHIN EACH BUNDLE.

        Args:
            boxes: Array of bounding boxes in format [x1, y1, x2, y2]

        Returns:
            Dictionary containing bundle information with simple ID mapping
        """
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

        # Calculate centers and sizes of all boxes
        centers, sizes = self._get_box_centers_and_sizes(boxes)

        # Calculate adaptive eps based on rebar sizes if enabled
        eps_to_use = self.eps
        if self.use_adaptive_eps:
            avg_size = np.mean(sizes)
            eps_to_use = max(self.eps, avg_size * 2.5)  # At least 2.5x average rebar size

        # Perform clustering using DBSCAN with adaptive eps
        clustering = DBSCAN(eps=eps_to_use, min_samples=self.min_samples).fit(centers)
        labels = clustering.labels_

        # Count rebars in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        bundles = []
        isolated_rebars = []
        id_mapping = {}
        unique_labels = set(labels)
        bundle_counter = 0

        # Identify clusters that are bundles (5+ rebars)
        bundle_clusters = []
        for label in unique_labels:
            if label != -1 and cluster_sizes[label] >= self.min_bundle_size:
                bundle_clusters.append(label)

        # Process bundles
        for label in bundle_clusters:
            cluster_indices = np.where(labels == label)[0]
            cluster_size = len(cluster_indices)
            bundle_counter += 1
            cluster_boxes = boxes[cluster_indices]
            cluster_centers = centers[cluster_indices]

            # Sort rebars within bundle row-wise then left-to-right
            sorted_indices = self._sort_rebars_rowwise_left_to_right(
                cluster_indices, cluster_centers
            )

            bundle_rebars = []
            for bundle_idx, global_idx in enumerate(sorted_indices, start=1):
                rebar_info = {
                    "global_index": int(global_idx),
                    "bundle_index": bundle_idx,
                    "display_id": bundle_idx,
                    "box": boxes[global_idx].tolist(),
                    "center": centers[global_idx].tolist(),
                    "row": self._get_row_number(centers[global_idx], cluster_centers),
                }
                bundle_rebars.append(rebar_info)
                id_mapping[int(global_idx)] = {
                    "display_id": bundle_idx,
                    "bundle_id": bundle_counter,
                    "bundle_index": bundle_idx,
                    "type": "bundle",
                    "group_size": cluster_size,
                }

            all_x1 = boxes[cluster_indices, 0]
            all_y1 = boxes[cluster_indices, 1]
            all_x2 = boxes[cluster_indices, 2]
            all_y2 = boxes[cluster_indices, 3]

            bundle_info = {
                "bundle_id": bundle_counter,
                "size": cluster_size,
                "rebars": bundle_rebars,
                "global_indices": sorted_indices.tolist(),
                "bounds": [
                    float(np.min(all_x1)),
                    float(np.min(all_y1)),
                    float(np.max(all_x2)),
                    float(np.max(all_y2)),
                ],
            }
            bundles.append(bundle_info)

        # Handle isolated rebars (those not in any bundle)
        all_processed_indices = set()
        for bundle in bundles:
            all_processed_indices.update(bundle["global_indices"])

        isolated_indices = [i for i in range(len(boxes)) if i not in all_processed_indices]

        if isolated_indices:
            isolated_centers = centers[isolated_indices]
            sorted_isolated = self._sort_rebars_rowwise_left_to_right(
                np.array(isolated_indices), isolated_centers
            )

            for display_idx, global_idx in enumerate(sorted_isolated, start=1):
                rebar_info = {
                    "global_index": int(global_idx),
                    "display_id": display_idx,
                    "box": boxes[global_idx].tolist(),
                    "center": centers[global_idx].tolist(),
                    "type": "isolated",
                    "group_size": 1,
                }
                isolated_rebars.append(rebar_info)
                id_mapping[int(global_idx)] = {
                    "display_id": display_idx,
                    "type": "isolated",
                    "group_size": 1,
                }

        total_rebars_in_bundles = sum(b["size"] for b in bundles)
        total_isolated = len(isolated_rebars)

        return {
            "bundles": bundles,
            "isolated_rebars": isolated_rebars,
            "total_bundles": len(bundles),
            "total_rebars_in_bundles": total_rebars_in_bundles,
            "total_isolated": total_isolated,
            "total_count": len(boxes),
            "id_mapping": id_mapping,
            "display_summary": {
                "total": len(boxes),
                "bundles": len(bundles),
                "rebars_in_bundles": total_rebars_in_bundles,
                "isolated": total_isolated,
            },
        }

    def _sort_rebars_rowwise_left_to_right(self, indices, centers):
        """
        Sort rebars by row (top to bottom) and left to right within each row
        """
        if len(indices) <= 1:
            return indices

        pairs = list(zip(indices, centers))
        pairs.sort(key=lambda x: x[1][1])  # sort by y

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
        """Determine which row this rebar belongs to within its cluster"""
        y_coords = sorted(set([c[1] for c in all_centers]))
        for i, y in enumerate(y_coords):
            if abs(center[1] - y) <= self.row_tolerance:
                return i + 1
        return 0

    def _get_box_centers_and_sizes(
        self, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate centers and approximate sizes of bounding boxes"""
        centers = []
        sizes = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])
            size = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            sizes.append(size)
        return np.array(centers), np.array(sizes)

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
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes[:, :4] /= r
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
        mn, mx = np.min(a), np.max(a)
        if 0.0 <= mn and mx <= 1.0:
            return a
        return self.sigmoid(a)

    def parse_onnx_outputs(self, outs):
        arrs = [np.asarray(o) for o in outs]
        # YOLO-NAS / ultralytics NMS-like outputs
        for a in arrs:
            if a.ndim == 3 and a.shape[-1] == 6:
                return "nms", (a[0] if a.shape[0] == 1 else a)
            if a.ndim == 2 and a.shape[-1] == 6:
                return "nms", a
        # "boxes + scores + classes" style
        num = None
        boxes = None
        scores = None
        classes = None
        for a in arrs:
            if a.ndim == 1 and a.size == 1:
                try:
                    num = int(a.reshape(-1)[0])
                except Exception:
                    pass
        for a in arrs:
            if a.ndim >= 2 and a.shape[-1] == 4:
                b = a[0] if a.ndim == 3 else a
                boxes = b.astype(np.float32)
        for a in arrs:
            if a.ndim >= 1 and a.shape[-1] != 4 and a.dtype.kind in "fc":
                s = a[0] if a.ndim == 2 else a
                if s.ndim == 1:
                    scores = s.astype(np.float32)
        for a in arrs:
            if a.ndim >= 1 and a.shape[-1] != 4:
                c = a[0] if a.ndim == 2 else a
                if c.ndim == 1:
                    classes = np.round(c).astype(np.int32)
        if boxes is not None and scores is not None and classes is not None:
            N = min(len(boxes), len(scores), len(classes))
            if num is not None and 0 < num <= N:
                N = num
            dets = np.zeros((N, 6), dtype=np.float32)
            dets[:, :4] = boxes[:N]
            dets[:, 4] = scores[:N]
            dets[:, 5] = classes[:N]
            return "nms", dets
        # Fallback: raw YOLO-style outputs [B, anchors, C]
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
        in_h, in_w = in_hw
        lb, r, dwdh = self.letterbox(image_bgr, (in_h, in_w))
        img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img, r, dwdh

    def draw_simple_black_frame(self, img, summary):
        """
        Draw a simple small black frame in top-left corner with white text
        """
        height, width = img.shape[:2]
        frame_width = 220
        frame_height = 140
        x1, y1 = 10, 10
        x2, y2 = x1 + frame_width, y1 + frame_height
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        title = "REBAR ANALYSIS"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img, title, (x1 + 15, y1 + 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        # separator
        cv2.line(img, (x1 + 10, y1 + 35), (x2 - 10, y1 + 35), (255, 255, 255), 1)
        y_offset = y1 + 50
        line_height = 18
        cv2.putText(
            img,
            f"TOTAL: {summary['total']}",
            (x1 + 15, y_offset),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += line_height
        cv2.putText(
            img,
            f"BUNDLES: {summary['bundles']}",
            (x1 + 15, y_offset),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += line_height
        cv2.putText(
            img,
            f"IN BUNDLES: {summary['rebars_in_bundles']}",
            (x1 + 15, y_offset),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += line_height
        cv2.putText(
            img,
            f"ISOLATED: {summary['isolated']}",
            (x1 + 15, y_offset),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img

    def draw_centered_ids_with_bundles(self, image_bgr, boxes, bundle_info=None):
        """
        Draw centered IDs on detected boxes, with optional bundle info overlay.
        """
        img = image_bgr.copy()
        img_height, img_width = img.shape[:2]

        # Draw all boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if bundle_info and bundle_info["total_count"] > 0:
            # Use bundle_info id_mapping
            for global_idx, id_info in bundle_info["id_mapping"].items():
                if global_idx < len(boxes):
                    box = boxes[global_idx]
                    x1, y1, x2, y2 = map(int, box[:4])
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_size = min(box_width, box_height)

                    if box_size < 25:
                        font_scale = 0.25
                        thickness = 1
                    elif box_size < 40:
                        font_scale = 0.3
                        thickness = 1
                    elif box_size < 70:
                        font_scale = 0.35
                        thickness = 1
                    elif box_size < 120:
                        font_scale = 0.4
                        thickness = 2
                    elif box_size < 200:
                        font_scale = 0.5
                        thickness = 2
                    else:
                        font_scale = 0.6
                        thickness = 2

                    display_id = str(id_info["display_id"])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_w, text_h), _ = cv2.getTextSize(
                        display_id, font, font_scale, thickness
                    )
                    padding = max(2, int(font_scale * 3))
                    bg_x1 = max(0, cx - text_w // 2 - padding)
                    bg_y1 = max(0, cy - text_h // 2 - padding)
                    bg_x2 = min(img_width, cx + text_w // 2 + padding)
                    bg_y2 = min(img_height, cy + text_h // 2 + padding)

                    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.putText(
                        img,
                        display_id,
                        (cx - text_w // 2, cy + text_h // 2),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
            img = self.draw_simple_black_frame(img, bundle_info["display_summary"])
        else:
            # If no bundles: simple numbering
            for idx, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = map(int, box[:4])
                box_size = min(x2 - x1, y2 - y1)

                if box_size < 30:
                    font_scale = 0.3
                    thickness = 1
                elif box_size < 60:
                    font_scale = 0.4
                    thickness = 1
                else:
                    font_scale = 0.5
                    thickness = 2

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                id_text = str(idx)
                (text_w, text_h), _ = cv2.getTextSize(
                    id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                cv2.rectangle(
                    img,
                    (cx - text_w // 2 - 2, cy - text_h // 2 - 2),
                    (cx + text_w // 2 + 2, cy + text_h // 2 + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    img,
                    id_text,
                    (cx - text_w // 2, cy + text_h // 2),
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

    def detect_rebars(
        self,
        image_bgr,
        model,
        class_id: int = 0,
        conf: float = 0.6,
        iou: float = 0.5,
        max_det: int = 10000,
    ):
        """
        Core ONNX-based detection logic.
        """
        try:
            sess = model["sess"]
            in_name = model["in_name"]
            in_hw = model["in_hw"]
            in_h, in_w = in_hw

            blob, r, dwdh = self.preprocess_for_onnx(image_bgr, in_hw)
            outs = sess.run(None, {in_name: blob})
            kind, data = self.parse_onnx_outputs(outs)

            dets_xyxy = []
            if kind == "nms":
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
            else:
                C = data.shape[1]
                boxes = data[:, :4].astype(np.float32)
                scores_mat_a = None
                scores_mat_b = None

                if C - 4 > 0:
                    scores_mat_a = self.maybe_sigmoid(data[:, 4:])
                if C - 5 > 0:
                    obj = self.maybe_sigmoid(data[:, 4:5])
                    cls_b = self.maybe_sigmoid(data[:, 5:])
                    scores_mat_b = obj * cls_b

                def count_above(m, t=0.25):
                    return int((m is not None) and (m.max(axis=1) >= t).sum())

                cnt_a = count_above(scores_mat_a)
                cnt_b = count_above(scores_mat_b)

                scores_mat = (
                    scores_mat_a
                    if (
                        scores_mat_a is not None
                        and (scores_mat_b is None or cnt_a >= cnt_b)
                    )
                    else scores_mat_b
                )

                if scores_mat is None or scores_mat.shape[1] <= class_id:
                    annotated = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    return (
                        annotated,
                        0,
                        "Model scores not found or class_id out of range.",
                        None,
                    )

                scores = scores_mat[:, class_id]

                if np.nanmax(boxes) <= 1.5:
                    boxes[:, [0, 2]] *= float(in_w)
                    boxes[:, [1, 3]] *= float(in_h)

                b_xyxy_from_xywh = self.xywh2xyxy(boxes)
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

            bundle_info = None
            if dets_xyxy:
                boxes_array = np.array(dets_xyxy)
                bundle_info = self.detect_bundles(boxes_array)
                annotated, sorted_boxes = self.draw_centered_ids_with_bundles(
                    image_bgr, boxes_array, bundle_info
                )
                count = len(sorted_boxes)
            else:
                annotated = image_bgr.copy()
                count = 0

            hd = self.to_hd_1080p(annotated, background=(18, 24, 31))
            banner_height = 100
            img_h, img_w, _ = hd.shape

            if bundle_info and bundle_info["total_bundles"] > 0:
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
            return (
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                0,
                f"Detection error: {e}",
                None,
            )

    # ------------------------------------------------------------------
    # FastAPI-oriented service methods (JSON responses)
    # ------------------------------------------------------------------
    def detect_image(self, file) -> Dict[str, Any]:
        """
        High-level API for FastAPI: detect on an uploaded image (bytes/file-like).
        """
        if hasattr(file, "read"):
            content = file.read()
        elif isinstance(file, (bytes, bytearray)):
            content = file
        else:
            raise ValueError(
                "Unsupported file type for detect_image; expected bytes or file-like object."
            )

        nparr = np.frombuffer(content, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return {
                "count": 0,
                "error": "Failed to decode image.",
                "bundle_info": None,
                "image": None,
            }

        annotated_rgb, count, error, bundle_info = self.detect_rebars(
            image_bgr, model
        )
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {
            "count": int(count),
            "error": error,
            "bundle_info": bundle_info,
            "image": image_data,
        }

    def detect_oak_camera(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        High-level API for FastAPI: detect on a frame from an OAK-D camera.
        """
        annotated_rgb, count, error, bundle_info = self.detect_rebars(
            frame_bgr, model
        )
        image_data = img_to_data_uri(annotated_rgb) if annotated_rgb is not None else None

        return {
            "count": int(count),
            "error": error,
            "bundle_info": bundle_info,
            "image": image_data,
        }

# Model loading (ONNX Runtime)
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
        raise ValueError(
            f"Expected a .onnx file, got: {os.path.splitext(model_path)[1]}"
        )

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

# Detection record helpers (DB) – PostgreSQL-style placeholders
def save_image_files(image_rgb: np.ndarray, det_id: str):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_path = os.path.join(DET_DIR, f"{det_id}.jpg")
    thumb_path = os.path.join(THUMB_DIR, f"{det_id}.jpg")

    cv2.imwrite(img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    h, w = img_bgr.shape[:2]
    tw = 360
    th = int(h * (tw / w))
    thumb = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
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
                "total_bundles": bundle_info["total_bundles"],
                "rebars_in_bundles": bundle_info["total_rebars_in_bundles"],
                "isolated": bundle_info["total_isolated"],
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
            count,
            w,
            h,
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
    cur.execute(
        "DELETE FROM detections WHERE id=%s AND user_id=%s",
        (det_id, user_id),
    )
    conn.commit()
    conn.close()
    return True

# Image → data URI helpers
def img_to_data_uri(
    image_rgb: np.ndarray, quality: int = 90, max_w: Optional[int] = None
) -> Optional[str]:
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