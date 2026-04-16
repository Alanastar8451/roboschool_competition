"""
YOLO-based object detector backend.

Detects only five allowed object categories and returns detections
for the mission finite-state machine.
"""

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
    _ALLOWED_CLASS_NAMES = {"cup", "laptop", "bottle", "chair", "backpack"}
    _ALIASES = {
        "cup": "cup",
        "mug": "cup",
        "laptop": "laptop",
        "notebook": "laptop",
        "bottle": "bottle",
        "water_bottle": "bottle",
        "chair": "chair",
        "backpack": "backpack",
    }

    def __init__(
        self,
        refs_dir: str,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.12,
        iou_threshold: float = 0.45,
        min_bbox_area: int = 400,
        imgsz: int = 640,
        device: str = "cpu",
        debug_visualization: bool = False,
    ):
        self.refs_dir = refs_dir
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_bbox_area = min_bbox_area
        self.imgsz = imgsz
        self.device = device
        self.debug_visualization = debug_visualization

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required. Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(self.model_path)
        self.model_names = self.model.names
        self.allowed_class_ids = self._resolve_allowed_class_ids()

    @classmethod
    def _canonical_name(cls, value: str) -> Optional[str]:
        key = value.strip().lower()
        if key in cls._ALIASES:
            return cls._ALIASES[key]
        if key in cls._ALLOWED_CLASS_NAMES:
            return key
        return None

    def _resolve_allowed_class_ids(self) -> List[int]:
        allowed = []
        if isinstance(self.model_names, dict):
            name_items = self.model_names.items()
        else:
            name_items = enumerate(self.model_names)

        for class_id, name in name_items:
            canonical = self._canonical_name(str(name))
            if canonical in self._ALLOWED_CLASS_NAMES:
                allowed.append(int(class_id))
        return allowed

    def detect(self, image_rgb: np.ndarray, target_name: str) -> List[Detection]:
        canonical_target = self._canonical_name(target_name)
        if canonical_target is None:
            return []
        if not self.allowed_class_ids:
            return []

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.allowed_class_ids,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        img_h, img_w = image_rgb.shape[:2]
        detections: List[Detection] = []
        for box in result.boxes:
            cls_tensor = box.cls
            conf_tensor = box.conf
            xyxy_tensor = box.xyxy
            if cls_tensor is None or conf_tensor is None or xyxy_tensor is None:
                continue

            class_id = int(cls_tensor.item())
            class_name = str(self.model_names[class_id])
            canonical = self._canonical_name(class_name)
            if canonical != canonical_target:
                continue

            x1, y1, x2, y2 = xyxy_tensor[0].cpu().numpy().tolist()
            x_min = max(0, min(int(x1), img_w))
            y_min = max(0, min(int(y1), img_h))
            x_max = max(0, min(int(x2), img_w))
            y_max = max(0, min(int(y2), img_h))

            if x_max <= x_min or y_max <= y_min:
                continue

            bbox_area = (x_max - x_min) * (y_max - y_min)
            if bbox_area < self.min_bbox_area:
                continue

            score = float(conf_tensor.item())
            detections.append(
                Detection(
                    class_name=canonical,
                    score=score,
                    bbox_xyxy=(x_min, y_min, x_max, y_max),
                    polygon_xy=None,
                    num_matches=0,
                    num_inliers=0,
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

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
