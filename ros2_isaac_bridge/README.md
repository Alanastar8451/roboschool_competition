# ROS 2 Jazzy Layer

Этот слой запускается отдельно от Isaac-контейнера и не ломает текущее окружение симуляции.

## 1) Поднять симуляцию

```bash
docker/ctl.sh up
docker/ctl.sh exec
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

## 2) Поднять ROS 2 Jazzy (rqt/bridge)

```bash
docker/ctl.sh ros2-build
docker/ctl.sh ros2-up
docker/ctl.sh ros2-exec
```

Внутри `ros2-jazzy` контейнера:

```bash
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh
```

## YOLO detector (5 classes only)

Mission node now uses YOLO and only accepts these categories:

- `cup`
- `laptop`
- `bottle`
- `chair`
- `backpack`

Install dependencies inside the `ros2-jazzy` container:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics
```

Run mission node (default model: `yolo11n.pt`):

```bash
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_mission_node.sh
```

Optional YOLO params:

```bash
ros2 run ros2_bridge_pkg mission_node --ros-args \
	-p yolo_model_path:=yolo11n.pt \
	-p yolo_device:=cpu \
	-p yolo_imgsz:=640 \
	-p yolo_iou_threshold:=0.45 \
	-p min_detection_score:=0.12
```

Во втором терминале (ещё один `docker/ctl.sh ros2-exec`):

```bash
rqt_graph
```

## Полезные команды в ROS 2 контейнере

```bash
ros2 topic list
ros2 topic echo /aliengo/base_velocity
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}" -r 10
```

Публикуемые топики bridge:

- `/aliengo/base_velocity` (`geometry_msgs/msg/TwistStamped`)
- `/aliengo/camera/color/image_raw` (`sensor_msgs/msg/Image`, `rgb8`)
- `/aliengo/camera/depth/image_raw` (`sensor_msgs/msg/Image`, `32FC1`)

## Полная E2E-проверка

Если нужен запуск всей цепочки одной командой, смотри:

```bash
bash /workspace/aliengo_competition/ros2_isaac_bridge/e2e_check/run_everything.sh
```
