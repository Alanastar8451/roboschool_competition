from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from aliengo_competition.common.run_logger import CompetitionRunLogger
from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState


_SPEED_LIMITS = {
    "vx": (-1.5, 1.5),
    "vy": (-0.75, 0.75),
    "wz": (-2.0, 2.0),
}

_OBJECT_ALIASES = {
    "cup": "cup",
    "mug": "cup",
    "laptop": "laptop",
    "notebook": "laptop",
    "bottle": "bottle",
    "water_bottle": "bottle",
    "chair": "chair",
    "backpack": "backpack",
}


def _canonical_object_name(name: str) -> Optional[str]:
    if name is None:
        return None
    return _OBJECT_ALIASES.get(str(name).strip().lower())


def _clip_speed(vx: float, vy: float, wz: float) -> Tuple[float, float, float]:
    vx_l, vx_h = _SPEED_LIMITS["vx"]
    vy_l, vy_h = _SPEED_LIMITS["vy"]
    wz_l, wz_h = _SPEED_LIMITS["wz"]
    return (
        float(np.clip(vx, vx_l, vx_h)),
        float(np.clip(vy, vy_l, vy_h)),
        float(np.clip(wz, wz_l, wz_h)),
    )


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "resources" / "assets" / "objects").is_dir():
            return parent
    return current.parent


def _build_reference_paths() -> Dict[str, Path]:
    root = _find_repo_root()
    base = root / "resources" / "assets" / "objects"
    candidates = {
        "cup": [base / "cup" / "cup.jpg", base / "cup" / "cup.png"],
        "laptop": [base / "laptop" / "laptop.png", base / "laptop" / "laptop.jpg"],
        "bottle": [base / "bottle" / "bottle.png", base / "bottle" / "bottle.jpg"],
        "chair": [base / "chair" / "chair.png", base / "chair" / "chair.jpg"],
        "backpack": [base / "backpack" / "backpack.jpg", base / "backpack" / "backpack.png"],
    }
    result: Dict[str, Path] = {}
    for key, options in candidates.items():
        for candidate in options:
            if candidate.is_file():
                result[key] = candidate
                break
    return result


@dataclass
class _TemplateData:
    gray: np.ndarray
    keypoints: Sequence
    descriptors: np.ndarray
    width: int
    height: int


@dataclass
class _TargetDetection:
    object_name: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]
    inliers: int
    matches: int
    center_x_px: float
    center_y_px: float


class _ReferenceFeatureDetector:
    def __init__(self):
        self._cv2 = None
        self._enabled = False
        self._templates: Dict[str, _TemplateData] = {}
        self._orb = None
        self._matcher = None
        self.min_good_matches = 14
        self.min_inliers = 9
        self.ratio_test = 0.78

        try:
            import cv2
        except Exception as exc:
            print(f"[Detector] cv2 недоступен: {exc}")
            return

        self._cv2 = cv2
        self._orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for object_name, path in _build_reference_paths().items():
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < self.min_inliers:
                continue
            h, w = gray.shape[:2]
            self._templates[object_name] = _TemplateData(
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                width=w,
                height=h,
            )

        self._enabled = bool(self._templates)
        if self._enabled:
            print(f"[Detector] Загружены шаблоны: {sorted(self._templates.keys())}")
        else:
            print("[Detector] Шаблоны не загружены, детекция отключена")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def detect(self, image_rgb: np.ndarray, target_name: str) -> Optional[_TargetDetection]:
        if not self._enabled or image_rgb is None:
            return None
        canonical_target = _canonical_object_name(target_name)
        if canonical_target is None:
            return None
        template = self._templates.get(canonical_target)
        if template is None:
            return None

        cv2 = self._cv2
        frame_rgb = np.asarray(image_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
            return None
        frame_rgb = frame_rgb[..., :3]
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        kp_frame, des_frame = self._orb.detectAndCompute(gray, None)
        if des_frame is None or len(kp_frame) < self.min_inliers:
            return None

        knn = self._matcher.knnMatch(template.descriptors, des_frame, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return None

        src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            return None

        inliers = int(mask.ravel().sum())
        if inliers < self.min_inliers:
            return None

        corners = np.float32(
            [
                [0.0, 0.0],
                [float(template.width - 1), 0.0],
                [float(template.width - 1), float(template.height - 1)],
                [0.0, float(template.height - 1)],
            ]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        h, w = gray.shape[:2]
        x_min = int(np.clip(np.floor(projected[:, 0].min()), 0, w - 1))
        y_min = int(np.clip(np.floor(projected[:, 1].min()), 0, h - 1))
        x_max = int(np.clip(np.ceil(projected[:, 0].max()), 0, w - 1))
        y_max = int(np.clip(np.ceil(projected[:, 1].max()), 0, h - 1))

        if x_max <= x_min or y_max <= y_min:
            return None
        if (x_max - x_min) * (y_max - y_min) < 350:
            return None

        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        score = float(inliers) / float(max(len(good), 1))
        return _TargetDetection(
            object_name=canonical_target,
            score=score,
            bbox_xyxy=(x_min, y_min, x_max, y_max),
            inliers=inliers,
            matches=len(good),
            center_x_px=center_x,
            center_y_px=center_y,
        )


def _safe_depth_value(depth_map: np.ndarray, cx: float, cy: float, win_radius: int = 6) -> Optional[float]:
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        return None

    h, w = depth.shape
    px = int(np.clip(round(cx), 0, w - 1))
    py = int(np.clip(round(cy), 0, h - 1))
    x0 = max(px - win_radius, 0)
    x1 = min(px + win_radius + 1, w)
    y0 = max(py - win_radius, 0)
    y1 = min(py + win_radius + 1, h)
    patch = depth[y0:y1, x0:x1]

    patch = patch[np.isfinite(patch)]
    patch = patch[(patch > 0.05) & (patch < 25.0)]
    if patch.size == 0:
        return None
    return float(np.percentile(patch, 35.0))


def _depth_center_clearance(depth_map: np.ndarray, width_frac: float = 0.35, height_frac: float = 0.3) -> Optional[float]:
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        return None
    h, w = depth.shape
    hw = int(0.5 * width_frac * w)
    hh = int(0.5 * height_frac * h)
    cx = w // 2
    cy = h // 2
    x0 = max(cx - hw, 0)
    x1 = min(cx + hw, w)
    y0 = max(cy - hh, 0)
    y1 = min(cy + hh, h)
    roi = depth[y0:y1, x0:x1]
    roi = roi[np.isfinite(roi)]
    roi = roi[(roi > 0.05) & (roi < 25.0)]
    if roi.size == 0:
        return None
    return float(np.percentile(roi, 20.0))


def _depth_center_min(depth_map: np.ndarray, width_frac: float = 0.42, height_frac: float = 0.42) -> Optional[float]:
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        return None
    h, w = depth.shape
    hw = int(0.5 * width_frac * w)
    hh = int(0.5 * height_frac * h)
    cx = w // 2
    cy = h // 2
    x0 = max(cx - hw, 0)
    x1 = min(cx + hw, w)
    y0 = max(cy - hh, 0)
    y1 = min(cy + hh, h)
    roi = depth[y0:y1, x0:x1]
    roi = roi[np.isfinite(roi)]
    roi = roi[(roi > 0.05) & (roi < 25.0)]
    if roi.size == 0:
        return None
    return float(np.min(roi))


def _depth_side_clearance(depth_map: np.ndarray, left: bool) -> Optional[float]:
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        return None
    h, w = depth.shape
    y0 = int(0.3 * h)
    y1 = int(0.9 * h)
    if left:
        x0, x1 = 0, int(0.45 * w)
    else:
        x0, x1 = int(0.55 * w), w
    roi = depth[y0:y1, x0:x1]
    roi = roi[np.isfinite(roi)]
    roi = roi[(roi > 0.05) & (roi < 25.0)]
    if roi.size == 0:
        return None
    return float(np.percentile(roi, 30.0))


def _bearing_from_pixel(center_x_px: float, image_width: int, h_fov_deg: float = 70.0) -> float:
    if image_width <= 1:
        return 0.0
    cx = 0.5 * float(image_width - 1)
    nx = (float(center_x_px) - cx) / max(cx, 1e-6)
    return float(nx * math.radians(0.5 * h_fov_deg))


def _normalize_object_queue(raw_queue) -> Sequence[Tuple[int, str]]:
    normalized = []
    if isinstance(raw_queue, (list, tuple)):
        for item in raw_queue:
            obj_id = None
            name = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                obj_id = item[0]
                name = item[1]
            elif isinstance(item, dict):
                obj_id = item.get("id")
                name = item.get("name")
            if obj_id is None or name is None:
                continue
            canonical = _canonical_object_name(str(name))
            if canonical is None:
                continue
            try:
                normalized.append((int(obj_id), canonical))
            except (TypeError, ValueError):
                continue
    return normalized


def _unwrap_env_from_robot(robot: AliengoRobotInterface):
    env = getattr(robot, "env", None)
    while env is not None and hasattr(env, "env") and getattr(env, "env") is not env:
        env = env.env
    return env


def _infer_control_dt(robot: AliengoRobotInterface, fallback_dt: float = 0.02) -> float:
    env = _unwrap_env_from_robot(robot)
    dt = getattr(env, "dt", None) if env is not None else None
    try:
        dt_value = float(dt)
        if dt_value > 0.0:
            return dt_value
    except (TypeError, ValueError):
        pass
    return float(fallback_dt)


class _CameraRenderer:
    def __init__(self, enabled: bool, depth_max_m: float):
        self.enabled = bool(enabled)
        self.depth_max_m = max(float(depth_max_m), 0.1)
        self._window_name = "Front Camera (Intel RealSense D435-like)"
        self._cv2 = None
        self._active = False
        if not self.enabled:
            return
        try:
            import cv2
        except Exception as exc:
            print(f"Отрисовка камеры отключена: не удалось импортировать cv2 ({exc})")
            self.enabled = False
            return
        self._cv2 = cv2
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)
        self._active = True

    def show(self, camera: CameraState) -> None:
        if not self._active or not isinstance(camera, CameraState):
            return
        image = camera.rgb
        depth = camera.depth
        if image is None or depth is None:
            return

        rgb = np.asarray(image)
        depth_m = np.asarray(depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_m.ndim != 2:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = rgb[..., :3]
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.depth_max_m)
        depth_u8 = (depth_m * (255.0 / self.depth_max_m)).astype(np.uint8)

        cv2 = self._cv2
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        view = np.concatenate((rgb_bgr, depth_color), axis=1)

        cv2.putText(view, "RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            view,
            f"Depth 0..{self.depth_max_m:.1f}m",
            (rgb.shape[1] + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self._window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if not self._active or self._cv2 is None:
            return
        self._cv2.destroyWindow(self._window_name)
        self._active = False


def run(
    robot: AliengoRobotInterface,
    steps: int = 15000,
    render_camera: bool = False,
    camera_depth_max_m: float = 4.0,
    seed: int = 0,
) -> None:
    robot.reset()
    env = getattr(robot, "env", None)
    if env is None:
        raise ValueError("Интерфейс робота должен предоставлять 'env' для обязательного логирования.")

    logger = CompetitionRunLogger(env=env, seed=int(seed))
    camera_renderer = _CameraRenderer(enabled=render_camera, depth_max_m=camera_depth_max_m)
    control_dt = _infer_control_dt(robot, fallback_dt=0.02)
    requested_steps = max(int(steps), 1)
    nominal_dt = 0.02
    target_duration_s = requested_steps * nominal_dt
    total_steps = max(int(round(target_duration_s / control_dt)), 1)
    print(
        f"[Контроллер] dt={control_dt:.4f}с, requested_steps={requested_steps}, "
        f"effective_steps={total_steps}"
    )
    object_queue = list(getattr(env, "SEQUENCE_OF_OBJECTS", []))
    sequence = list(_normalize_object_queue(object_queue))
    if not sequence:
        object_id_to_name = getattr(env, "object_id_to_name", {})
        for object_id in sorted(object_id_to_name.keys()):
            canonical = _canonical_object_name(str(object_id_to_name[object_id]))
            if canonical is not None:
                sequence.append((int(object_id), canonical))

    print(f"[Контроллер] отрисовка_камеры={'включена' if camera_renderer.enabled else 'выключена'}")
    print(f"[Контроллер] object_queue={object_queue}")
    print(f"[Контроллер] normalized_sequence={sequence}")

    detector = _ReferenceFeatureDetector()

    # Редактируемые пользователем блоки в этом файле:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    search_vx = 0.30
    search_vy_amp = 0.08
    search_wz_base = 0.95
    search_period_s = 12.0
    reacquire_timeout_s = 1.6

    approach_vx_max = 0.62
    approach_distance_target_m = 0.52
    approach_slowdown_distance_m = 2.20
    stop_distance_m = 0.62
    success_depth_m = 0.80
    success_bearing_rad = math.radians(8.0)
    success_hold_frames = 8

    obstacle_emergency_dist_m = 0.78
    obstacle_slowdown_dist_m = 1.15

    stuck_speed_eps = 0.05
    stuck_command_min_vx = 0.22
    avoid_reverse_vx = -0.22
    avoid_turn_wz = 1.65
    avoid_forward_vx = 0.07
    avoid_reverse_s = 0.55
    avoid_total_s = 1.45
    # ================== USER PARAMETERS END ==================

    target_idx = 0
    hold_counter = 0
    last_seen_t = -1e9
    last_seen_bearing = 0.0
    logged_object_ids = set()
    avoid_mode_end_t = -1e9
    avoid_reverse_end_t = -1e9
    avoid_turn_sign = 1.0

    try:
        initial_observation = robot.get_observation()
        initial_camera_payload = robot.get_camera()
        print(
            "[Контроллер] Предпросмотр API:"
            f" observation_type={type(initial_observation).__name__},"
            f" camera_payload={'да' if initial_camera_payload is not None else 'нет'}"
        )
        if initial_camera_payload is None:
            print(
                "[Контроллер] Предупреждение: данные фронтальной камеры недоступны. "
                "Проверьте, что симулятор не запущен в headless-режиме и что включён front_camera_enabled."
            )

        for step_index in range(total_steps):
            state = robot.get_state()

            # Камеру можно брать и из state, и напрямую через robot.get_camera().
            camera_payload = robot.get_camera()
            camera_state = state.camera
            if (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, dict):
                camera_state = CameraState(
                    rgb=camera_payload.get("image"),
                    depth=camera_payload.get("depth"),
                )
            elif (camera_state.rgb is None or camera_state.depth is None) and isinstance(camera_payload, CameraState):
                camera_state = camera_payload
            camera_renderer.show(camera_state)

            # ================= USER CONTROL LOGIC START =================
            # Это основной блок для логики участника.
            # Здесь нужно читать измерения, принимать решение и формировать
            # команды движения. Логирование найденного объекта тоже делается
            # отсюда.
            #
            # Формат данных эквивалентен данных:
            # - вход команды: vx, vy, wz
            # - выход состояния: measured_vx, measured_vy, measured_wz
            # - joint_states: joint_names, relative_dof_pos, dof_vel
            # - imu: base_ang_vel, base_lin_acc
            # - camera: camera_data["image"], camera_data["depth"]
            # - порядок объектов: object_queue
            #
            # Ниже приведён обязательный шаблон. Участник должен:
            # 1. реализовать get_found_object_id(...)
            # 2. при обнаружении объекта вернуть его id
            # 3. обязательно вызвать log_found_object(...)
            #
            # Если объект не найден, верните None.
            sim_t = state.sim_time_s

            joint_names = state.joints.name
            relative_dof_pos = state.q
            dof_vel = state.q_dot
            measured_vx = state.vx
            measured_vy = state.vy
            measured_wz = state.wz
            base_ang_vel = state.imu.angular_velocity_xyz
            base_lin_acc = np.zeros(3, dtype=np.float32)
            camera_data = camera_payload if isinstance(camera_payload, dict) else {
                "image": camera_state.rgb,
                "depth": camera_state.depth,
            }

            # Обязательная обвязка для логирования найденного объекта.
            # Использование:
            # - get_found_object_id(...) должен вернуть id найденного объекта
            #   или None, если объект не найден
            # - log_found_object(...) записывает событие в судейский лог
            # - этот шаблон нельзя удалять: участник обязан реализовать его
            #   в своём решении
            #
            # Пример:
            #     detected_object_id = get_found_object_id(...)
            #     if detected_object_id is not None:
            #         log_found_object(detected_object_id)
            def log_found_object(object_id: int) -> None:
                """ОБЯЗАТЕЛЬНО: вызывайте при обнаружении целевого объекта."""
                logger.log_detected_object_at_time(int(object_id), float(sim_t))

            def get_found_object_id(
                current_state,
                current_camera_data,
                current_object_queue,
            ):
                """Возвращает id текущего целевого объекта после уверенного достижения."""
                nonlocal target_idx
                nonlocal hold_counter
                nonlocal last_seen_t
                nonlocal last_seen_bearing

                if target_idx >= len(sequence):
                    return None

                target_id, target_name = sequence[target_idx]
                image = current_camera_data.get("image") if isinstance(current_camera_data, dict) else None
                depth = current_camera_data.get("depth") if isinstance(current_camera_data, dict) else None
                detection = detector.detect(image, target_name) if detector.enabled else None
                if detection is None:
                    hold_counter = 0
                    return None

                img_h, img_w = np.asarray(image).shape[:2]
                depth_h, depth_w = np.asarray(depth).shape[:2] if depth is not None else (0, 0)
                if depth_h <= 0 or depth_w <= 0 or img_h <= 0 or img_w <= 0:
                    hold_counter = 0
                    return None

                depth_x = detection.center_x_px * (float(depth_w) / float(img_w))
                depth_y = detection.center_y_px * (float(depth_h) / float(img_h))
                depth_m = _safe_depth_value(depth, depth_x, depth_y)
                if depth_m is None:
                    hold_counter = 0
                    return None

                bearing_rad = _bearing_from_pixel(detection.center_x_px, img_w, h_fov_deg=70.0)
                last_seen_t = sim_t
                last_seen_bearing = bearing_rad

                if depth_m <= success_depth_m and abs(bearing_rad) <= success_bearing_rad:
                    hold_counter += 1
                else:
                    hold_counter = 0

                if hold_counter >= success_hold_frames:
                    hold_counter = 0
                    target_idx += 1
                    print(
                        f"[Контроллер] Цель достигнута: id={target_id}, name={target_name}, "
                        f"t={sim_t:.2f}s, depth={depth_m:.2f}m"
                    )
                    return int(target_id)
                return None

            detected_object_id = get_found_object_id(
                state,
                camera_data,
                object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)
                logged_object_ids.add(int(detected_object_id))

            if target_idx >= len(sequence):
                vx = 0.0
                vy = 0.0
                vw = 0.0
                if step_index % max(int(round(1.0 / max(control_dt, 1e-3))), 1) == 0:
                    print(f"[Контроллер] Все цели достигнуты: {sorted(logged_object_ids)}")
            else:
                target_id, target_name = sequence[target_idx]
                image = camera_data.get("image") if isinstance(camera_data, dict) else None
                depth = camera_data.get("depth") if isinstance(camera_data, dict) else None
                detection = detector.detect(image, target_name) if detector.enabled else None

                if detection is not None and image is not None and depth is not None:
                    img_h, img_w = np.asarray(image).shape[:2]
                    depth_h, depth_w = np.asarray(depth).shape[:2]
                    depth_x = detection.center_x_px * (float(depth_w) / float(max(img_w, 1)))
                    depth_y = detection.center_y_px * (float(depth_h) / float(max(img_h, 1)))
                    depth_m = _safe_depth_value(depth, depth_x, depth_y)
                    bearing = _bearing_from_pixel(detection.center_x_px, img_w, h_fov_deg=70.0)
                    if depth_m is not None:
                        last_seen_t = sim_t
                        last_seen_bearing = bearing

                        depth_error = max(depth_m - approach_distance_target_m, 0.0)
                        depth_gain = min(depth_error / max(approach_slowdown_distance_m, 1e-3), 1.0)
                        align_gain = max(0.0, 1.0 - abs(bearing) / math.radians(40.0))

                        vx = approach_vx_max * depth_gain * align_gain
                        vy = 0.0
                        vw = 2.2 * bearing - 0.25 * measured_wz
                        if depth_m < stop_distance_m:
                            vx *= 0.2
                    else:
                        detection = None

                if detection is None:
                    time_since_seen = sim_t - last_seen_t
                    if time_since_seen <= reacquire_timeout_s:
                        vx = 0.14
                        vy = 0.0
                        vw = np.sign(last_seen_bearing) * 1.0
                    else:
                        phase = 2.0 * math.pi * sim_t / max(search_period_s, control_dt)
                        direction = 1.0 if int(sim_t / max(0.5 * search_period_s, 1.0)) % 2 == 0 else -1.0
                        vx = search_vx * (0.75 + 0.25 * math.cos(phase))
                        vy = search_vy_amp * math.sin(0.5 * phase)
                        vw = direction * (search_wz_base + 0.35 * math.sin(phase))

                center_clearance = _depth_center_clearance(depth)
                center_min_clearance = _depth_center_min(depth)
                left_clearance = _depth_side_clearance(depth, left=True)
                right_clearance = _depth_side_clearance(depth, left=False)

                emergency_collision_risk = (
                    (center_min_clearance is not None and center_min_clearance < obstacle_emergency_dist_m)
                    or (center_clearance is not None and center_clearance < 0.9 * obstacle_emergency_dist_m)
                )
                likely_stuck = (
                    abs(measured_vx) < stuck_speed_eps
                    and abs(vx) > stuck_command_min_vx
                    and center_clearance is not None
                    and center_clearance < obstacle_slowdown_dist_m
                )

                if emergency_collision_risk or likely_stuck:
                    turn_sign = avoid_turn_sign
                    if left_clearance is not None and right_clearance is not None:
                        turn_sign = 1.0 if left_clearance >= right_clearance else -1.0
                    avoid_turn_sign = turn_sign
                    avoid_reverse_end_t = max(avoid_reverse_end_t, sim_t + avoid_reverse_s)
                    avoid_mode_end_t = max(avoid_mode_end_t, sim_t + avoid_total_s)
                elif center_clearance is not None and center_clearance < obstacle_slowdown_dist_m:
                    vx *= 0.55

                if sim_t < avoid_mode_end_t:
                    if sim_t < avoid_reverse_end_t:
                        vx = avoid_reverse_vx
                        vy = 0.0
                        vw = 0.0
                    else:
                        vx = avoid_forward_vx
                        vy = 0.0
                        vw = avoid_turn_wz * avoid_turn_sign

                if step_index % max(int(round(1.0 / max(control_dt, 1e-3))), 1) == 0:
                    status = f"target={target_id}:{target_name}, seen={'yes' if detection is not None else 'no'}"
                    if center_clearance is not None:
                        status += f", clearance={center_clearance:.2f}m"
                    if center_min_clearance is not None:
                        status += f", min_clearance={center_min_clearance:.2f}m"
                    if sim_t < avoid_mode_end_t:
                        status += ", recovery=active"
                    print(f"[Контроллер] {status}")

            vx, vy, vw = _clip_speed(vx, vy, vw)
            # ================== USER CONTROL LOGIC END ==================

            robot.set_speed(vx, vy, vw)
            robot.step()
            logger.log_step(step_index * control_dt)
            robot.get_observation()  # Пример доступа к наблюдению после step().

            if robot.is_fallen():
                robot.stop()
                robot.reset()
                hold_counter = 0
                last_seen_t = -1e9
                avoid_mode_end_t = -1e9
                avoid_reverse_end_t = -1e9
                print("[Контроллер] робот упал -> сброс")
                continue
    finally:
        logger.close()
        camera_renderer.close()
        robot.stop()
