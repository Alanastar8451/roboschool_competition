"""
AKAZE-based reference image detector.

Matches a reference texture image against the current camera frame
using local feature descriptors and homography estimation.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    class_name: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]
    polygon_xy: Optional[List[List[float]]] = None
    num_matches: int = 0
    num_inliers: int = 0


class ReferenceImageDetectorBackend:
    def __init__(
        self,
        refs_dir: str,
        min_matches: int = 8,
        min_inliers: int = 6,
        ransac_reproj_threshold: float = 5.0,
        min_bbox_area: int = 400,
        lowe_ratio: float = 0.75,
        debug_visualization: bool = False,
    ):
        self.refs_dir = refs_dir
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_bbox_area = min_bbox_area
        self.lowe_ratio = lowe_ratio
        self.debug_visualization = debug_visualization

        self.akaze = cv2.AKAZE_create()
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._ref_cache: dict = {}

    # ------------------------------------------------------------------
    # Reference image loading with caching
    # ------------------------------------------------------------------
    def _find_ref_path(self, target_name: str) -> Optional[str]:
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self.refs_dir, f"{target_name}{ext}")
            if os.path.isfile(path):
                return path
            path = os.path.join(self.refs_dir, target_name, f"{target_name}{ext}")
            if os.path.isfile(path):
                return path
        return None

    def _load_ref(self, target_name: str):
        if target_name in self._ref_cache:
            return self._ref_cache[target_name]

        path = self._find_ref_path(target_name)
        if path is None:
            self._ref_cache[target_name] = None
            return None

        img = cv2.imread(path)
        if img is None:
            self._ref_cache[target_name] = None
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, desc = self.akaze.detectAndCompute(gray, None)
        if desc is None or len(kp) < 4:
            kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None or len(kp) < 4:
            self._ref_cache[target_name] = None
            return None

        entry = {
            "kp": kp,
            "desc": desc,
            "shape": img.shape[:2],
            "path": path,
        }
        self._ref_cache[target_name] = entry
        return entry

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, image_rgb: np.ndarray, target_name: str) -> List[Detection]:
        ref = self._load_ref(target_name)
        if ref is None:
            return []

        ref_kp = ref["kp"]
        ref_desc = ref["desc"]
        ref_h, ref_w = ref["shape"]

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kp, desc = self.akaze.detectAndCompute(gray, None)

        if desc is None or len(kp) < self.min_matches:
            kp, desc = self.orb.detectAndCompute(gray, None)
            if desc is None or len(kp) < self.min_matches:
                return []

        if ref_desc.dtype != desc.dtype:
            return []

        try:
            raw_matches = self.bf_hamming.knnMatch(ref_desc, desc, k=2)
        except cv2.error:
            return []

        good = []
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.lowe_ratio * n.distance:
                    good.append(m)

        if len(good) < self.min_matches:
            return []

        src_pts = np.float32(
            [ref_kp[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold
        )
        if H is None or mask is None:
            return []

        num_inliers = int(mask.sum())
        if num_inliers < self.min_inliers:
            return []

        corners_ref = np.float32(
            [[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]
        ).reshape(-1, 1, 2)

        try:
            projected = cv2.perspectiveTransform(corners_ref, H)
        except cv2.error:
            return []

        polygon = projected.reshape(-1, 2)

        poly_area = cv2.contourArea(polygon.astype(np.float32))
        if poly_area < self.min_bbox_area:
            return []

        if not _is_convex_enough(polygon):
            return []

        img_h, img_w = image_rgb.shape[:2]
        x_min = max(0, int(polygon[:, 0].min()))
        y_min = max(0, int(polygon[:, 1].min()))
        x_max = min(img_w, int(polygon[:, 0].max()))
        y_max = min(img_h, int(polygon[:, 1].max()))

        bbox_area = (x_max - x_min) * (y_max - y_min)
        if bbox_area < self.min_bbox_area:
            return []

        img_area = img_h * img_w
        if bbox_area > img_area * 0.55:
            return []

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w > img_w * 0.80 or bbox_h > img_h * 0.80:
            return []
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
        if aspect > 5.0:
            return []

        inlier_ratio = num_inliers / max(len(good), 1)
        normalized = min(1.0, num_inliers / 20.0)
        score = inlier_ratio * normalized

        det = Detection(
            class_name=target_name,
            score=score,
            bbox_xyxy=(x_min, y_min, x_max, y_max),
            polygon_xy=polygon.tolist(),
            num_matches=len(good),
            num_inliers=num_inliers,
        )
        return [det]

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------
    def draw_debug(
        self, image_rgb: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        vis = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if det.polygon_xy is not None:
                pts = np.array(det.polygon_xy, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (255, 0, 0), 2)

            label = (
                f"{det.class_name} s={det.score:.2f} "
                f"m={det.num_matches} i={det.num_inliers}"
            )
            cv2.putText(
                vis, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return vis


def _is_convex_enough(polygon: np.ndarray) -> bool:
    """Reject self-intersecting quadrilaterals from bad homographies."""
    n = len(polygon)
    if n < 3:
        return False
    cross_signs = []
    for i in range(n):
        a = polygon[(i + 1) % n] - polygon[i]
        b = polygon[(i + 2) % n] - polygon[(i + 1) % n]
        cross = a[0] * b[1] - a[1] * b[0]
        if abs(cross) > 1e-6:
            cross_signs.append(cross > 0)
    if not cross_signs:
        return False
    return all(s == cross_signs[0] for s in cross_signs)
