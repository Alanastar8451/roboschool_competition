from __future__ import annotations

import math

import numpy as np

import time

from aliengo_competition.common.run_logger import CompetitionRunLogger
from aliengo_competition.robot_interface.base import AliengoRobotInterface
from aliengo_competition.robot_interface.types import CameraState


_YOLO_MODEL = None
_YOLO_MODEL_LOAD_ERROR = None
_YOLO_DEBUG = False


def _yolo_dbg(msg: str) -> None:
    if _YOLO_DEBUG:
        print(f"[YOLO] {msg}")


def _get_yolo_model():
    global _YOLO_MODEL, _YOLO_MODEL_LOAD_ERROR
    if _YOLO_MODEL is not None:
        _yolo_dbg("model: cached")
        return _YOLO_MODEL
    if _YOLO_MODEL_LOAD_ERROR is not None:
        _yolo_dbg(f"model: previously failed to load ({_YOLO_MODEL_LOAD_ERROR})")
        return None
    try:
        from ultralytics import YOLO

        _yolo_dbg("model: loading yolov8n.pt")
        _YOLO_MODEL = YOLO("yolo26n.pt")
        _yolo_dbg("model: loaded ok")
        return _YOLO_MODEL
    except Exception as exc:
        _YOLO_MODEL_LOAD_ERROR = exc
        _yolo_dbg(f"model: load failed ({exc})")
        return None


box = None
FLAG = 'walk'
timestamp = None

def get_found_object_id(
    current_state,
    current_camera_data,
    current_object_queue,
):
    """
    Detect the *next* target object from the queue using YOLO and return its id.

    Only these object types are considered: mug, laptop, bottle, chair, backpack.
    In the simulator mapping "mug" corresponds to COCO class "cup".
    """

    global box

    # Queue items are typically (id, name). We will detect ANY of them.
    if not current_object_queue:
        _yolo_dbg("return None: object_queue is empty")
        return None

    # Restrict to allowed classes. "mug" is treated as "cup".
    allowed_names = {"backpack", "bottle", "chair", "laptop", "mug", "cup"}

    # Build a map: yolo_class_name -> earliest queue index and its object id
    queue_map: dict[str, tuple[int, int]] = {}
    for qi, item in enumerate(current_object_queue):
        if not (isinstance(item, (tuple, list)) and len(item) >= 2):
            _yolo_dbg(f"skip queue item: unexpected shape ({type(item).__name__}): {item}")
            continue
        try:
            obj_id = int(item[0])
        except Exception:
            _yolo_dbg(f"skip queue item: cannot int() id: {item}")
            continue
        obj_name = str(item[1]).strip().lower()
        if obj_name not in allowed_names:
            continue
        yolo_name = "cup" if obj_name == "mug" else obj_name
        if yolo_name not in queue_map or qi < queue_map[yolo_name][0]:
            queue_map[yolo_name] = (qi, obj_id)

    if not queue_map:
        _yolo_dbg(f"return None: no allowed objects in queue (allowed={sorted(allowed_names)})")
        return None
    _yolo_dbg(f"queue targets: { {k: v[1] for k, v in queue_map.items()} }")

    image = None
    if isinstance(current_camera_data, dict):
        image = current_camera_data.get("image")
    if image is None:
        _yolo_dbg(
            f"return None: camera image is None (camera_data keys={list(current_camera_data.keys()) if isinstance(current_camera_data, dict) else type(current_camera_data).__name__})"
        )
        return None

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        _yolo_dbg(f"return None: bad rgb shape={getattr(rgb, 'shape', None)}")
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = rgb[..., :3]

    model = _get_yolo_model()
    if model is None:
        _yolo_dbg("return None: model is None (not installed / failed load)")
        return None

    try:
        _yolo_dbg(f"predict: start (shape={rgb.shape}, dtype={rgb.dtype})")
        results = model.predict(source=rgb, verbose=False, conf=0.3)
    except Exception as exc:
        _yolo_dbg(f"return None: predict exception ({exc})")
        return None
    if not results:
        _yolo_dbg("return None: predict returned empty results")
        return None

    result0 = results[0]
    names = getattr(result0, "names", None) or {}
    boxes = getattr(result0, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        _yolo_dbg(f"return None: no boxes/cls (boxes={type(boxes).__name__ if boxes is not None else None})")
        return None

    classes = boxes.cls
    confs = getattr(boxes, "conf", None)
    xyxy = getattr(boxes, "xyxy", None)
    try:
        classes_np = classes.detach().cpu().numpy() if hasattr(classes, "detach") else np.asarray(classes)
        confs_np = (
            confs.detach().cpu().numpy() if (confs is not None and hasattr(confs, "detach")) else np.asarray(confs)
            if confs is not None
            else None
        )
        xyxy_np = (
            xyxy.detach().cpu().numpy() if (xyxy is not None and hasattr(xyxy, "detach")) else np.asarray(xyxy)
            if xyxy is not None
            else None
        )
    except Exception as exc:
        _yolo_dbg(f"return None: failed to convert tensors to numpy ({exc})")
        return None

    _yolo_dbg(f"predict: got {len(classes_np)} detections")

    # Determine best match among detections:
    # - prefer earliest object in queue
    # - tie-breaker: highest confidence
    best = None  # (queue_index, -conf, object_id, cls_name, det_index)
    for idx, cls_id in enumerate(classes_np.tolist()):
        cls_name = str(names.get(int(cls_id), "")).lower()
        cls_conf = float(confs_np[idx]) if confs_np is not None else None
        _yolo_dbg(f"det[{idx}]: cls_id={int(cls_id)} name='{cls_name}' conf={cls_conf}")
        if cls_name not in queue_map:
            continue
        if confs_np is not None and float(confs_np[idx]) < 0.4:
            _yolo_dbg(f"match but low conf: {float(confs_np[idx]):.3f} < 0.4")
            continue
        queue_index, obj_id = queue_map[cls_name]
        conf_value = float(confs_np[idx]) if confs_np is not None else 1.0
        candidate = (queue_index, -conf_value, obj_id, cls_name, idx)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        _yolo_dbg("return None: none of the allowed queue objects found in detections")
        print("return None: none of the allowed queue objects found in detections")
        return None

    _queue_index, _neg_conf, obj_id, cls_name, det_index = best
    box = None
    if xyxy_np is not None and len(xyxy_np) > det_index:
        try:
            x1, y1, x2, y2 = [float(v) for v in xyxy_np[det_index].tolist()]
            conf_value = float(confs_np[det_index]) if confs_np is not None else None
            box = (x1, y1, x2, y2, conf_value, cls_name, int(obj_id))
            _yolo_dbg(f"box: {box}")
            print(f"box: {box}")
        except Exception as exc:
            box = None
            _yolo_dbg(f"box: failed to extract ({exc})")
            print(f"box: failed to extract ({exc})")
    else:
        _yolo_dbg("box: not available (xyxy missing)")
        print("box: not available (xyxy missing)")
    _yolo_dbg(f"FOUND: class='{cls_name}' returning object_id={obj_id}")
    print(f"FOUND: class='{cls_name}' returning object_id={obj_id}")
    return obj_id


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
    global FLAG, box, timestamp

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
    print(f"[Контроллер] отрисовка_камеры={'включена' if camera_renderer.enabled else 'выключена'}")
    print(f"[Контроллер] object_queue={object_queue}")

    # Редактируемые пользователем блоки в этом файле:
    # 1. USER PARAMETERS START / END
    # 2. USER CONTROL LOGIC START / END

    # ================= USER PARAMETERS START =================
    # Настраивайте эти значения, чтобы менять поведение демо.
    # Параметры, завязанные на время, пересчитываются через шаг симуляции, потому время в секундах работают в симуляции правильно
    warmup_s = 0.4
    ramp_s = 1.2
    trajectory_period_s = 8.0
    forward_speed_mean = 0.40
    forward_speed_amp = 0.35
    lateral_speed_amp = 0.22
    yaw_rate_amp = 0.75
    yaw_rate_damping = 0.55
    ang_vel_scale = 0.25
    # ================== USER PARAMETERS END ==================

    segment_start_t = 0.0

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
            omega_z = state.imu.wz / ang_vel_scale

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

            detected_object_id = get_found_object_id(
                state,
                camera_data,
                object_queue,
            )
            if detected_object_id is not None:
                log_found_object(detected_object_id)

            if detected_object_id is not None and detected_object_id == object_queue[0][0]:
                FLAG = 'object_found'
#                 print(f"""\n\n[IMPORTANT]
# [IMPORTANT] detected_object_id={detected_object_id} == object_queue[0]={object_queue[0]}
# [IMPORTANT]\n\n""")
#                 object_queue.pop(0)
#                 time.sleep(2)

            local_t = max(sim_t - segment_start_t, 0.0)
            if local_t < warmup_s:
                vx = 0.0
                vy = 0.0
                vw = 0.0
            else:
                motion_t = local_t - warmup_s
                phase = 2.0 * math.pi * motion_t / max(trajectory_period_s, control_dt)
                ramp = min(motion_t / max(ramp_s, control_dt), 1.0)

                vx = ramp * (forward_speed_mean + forward_speed_amp * math.cos(phase))
                vy = ramp * (lateral_speed_amp * math.sin(2.0 * phase))
                yaw_ff = yaw_rate_amp * math.sin(phase + math.pi / 4.0)
                vw = ramp * (yaw_ff - yaw_rate_damping * state.imu.wz / ang_vel_scale)
                vw = max(min(vw, 1.0), -1.0)

            center_x = camera_data["depth"].shape[1] // 2
            center_y = camera_data["depth"].shape[0] // 2

            if FLAG == 'walk':
                if camera_data["depth"][center_y, center_x] > 1 and camera_data["depth"][center_y, center_x // 2] > 1:
                    vx = 1.5
                else:
                    vw = -2.0
            elif FLAG == 'object_found':
                if box is not None:
                    x1, y1, x2, y2, conf_value, cls_name, obj_id = box
                    object_center_x = (x1 + x2) / 2

                    if object_center_x > center_x:
                        vw = -1.0
                        vx = 1.5
                    else:
                        vw = 1.0
                        vx = 1.5

                    if camera_data["depth"][center_y, center_x] < 0.75 and detected_object_id == object_queue[0][0]:
                        print(f"""\n\n[IMPORTANT]
# [IMPORTANT] detected_object_id={detected_object_id} == object_queue[0]={object_queue[0]}
# [IMPORTANT]\n\n""")
                        object_queue.pop(0)
                        timestamp = time.time()
                        FLAG = 'wait'
                else:
                    FLAG = 'walk'
            elif FLAG == 'wait':
                vx = 0.0
                vy = 0.0
                vw = 0.0
                if time.time() - timestamp > 2:
                    FLAG = 'walk'
            # ================== USER CONTROL LOGIC END ==================

            robot.set_speed(vx, vy, vw)
            robot.step()
            logger.log_step(step_index * control_dt)
            robot.get_observation()  # Пример доступа к наблюдению после step().

            if robot.is_fallen():
                robot.stop()
                robot.reset()
                segment_start_t = (step_index + 1) * control_dt
                print("[Контроллер] робот упал -> сброс")
                continue
    finally:
        logger.close()
        camera_renderer.close()
        robot.stop()
