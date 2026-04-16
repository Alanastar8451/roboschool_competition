import time
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Маппинг классов YOLO -> id миссии
CLASS_TO_OBJ_ID = {
    "backpack": 24,
    "chair": 56,
    "bottle": 39,
    "laptop": 63,
    "cup": 41,
}


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()

        # модель — поменяйте путь на вашу модель или weight
        self.model_path = 'yolo26m.pt'
        self.model_conf = 0.35
        self.model_iou = 0.45
        self.device = 'cuda'  # поменяйте на 'cuda' если доступно

        if YOLO is None:
            self.get_logger().error('ultralytics.YOLO не установлен. Установите пакет ultralytics.')
            self.model = None
        else:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                self.get_logger().error(f'Ошибка загрузки модели {self.model_path}: {e}')
                self.model = None

        # подписки на RGB и DEPTH топики, которые публикует bridge_node
        self.create_subscription(Image, '/aliengo/camera/color/image_raw', self.rgb_cb, 5)
        self.create_subscription(Image, '/aliengo/camera/depth/image_raw', self.depth_cb, 5)

        # публикуем найденный id как Int32 -> bridge ожидает '/aliengo/detected_object_id'
        self.detect_pub = self.create_publisher(Int32, '/aliengo/detected_object_id', 10)

        # debug image
        self.debug_pub = self.create_publisher(Image, '/aliengo/detections/image', 1)

        self.last_depth = None
        self.depth_ts = 0.0

        # параметры камеры (совпадают с isaac_controller.py по умолчанию)
        self.declare_parameter('color_width', 640)
        self.declare_parameter('color_height', 360)
        self.declare_parameter('color_fov_deg', 70.0)

        self.get_logger().info('YOLO26 Object Detector запущен')

    def depth_cb(self, msg: Image):
        try:
            if msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding in ('16UC1', '16U'):
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32)
                depth /= 1000.0
            else:
                # попытка интерпретировать как float32
                depth = np.frombuffer(msg.data, dtype=np.float32)
                try:
                    depth = depth.reshape((msg.height, msg.width))
                except Exception:
                    return
        except Exception as e:
            self.get_logger().error(f'depth_cb error: {e}')
            return

        self.last_depth = depth
        self.depth_ts = time.time()

    def rgb_cb(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            self.get_logger().error(f'CVBridge rgb error: {e}')
            return

        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        detections = []

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=img_bgr,
                    conf=self.model_conf,
                    iou=self.model_iou,
                    device=self.device,
                    verbose=False,
                    stream=False,
                )
            except Exception as e:
                self.get_logger().error(f'model.predict error: {e}')
                results = []

            for r in results:
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                # boxes.xyxy, boxes.cls, boxes.conf
                for i in range(len(boxes)):
                    try:
                        xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else np.array(boxes.xyxy[i])
                        cls_id = int(boxes.cls[i]) if hasattr(boxes, 'cls') else int(boxes.data[i, 5])
                        conf = float(boxes.conf[i]) if hasattr(boxes, 'conf') else float(boxes.data[i, 4])
                        class_name = r.names[cls_id]
                    except Exception:
                        continue

                    if class_name not in CLASS_TO_OBJ_ID:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy[:4])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    z = None
                    if self.last_depth is not None and time.time() - self.depth_ts < 0.5:
                        y1c = max(0, y1)
                        y2c = min(self.last_depth.shape[0], y2)
                        x1c = max(0, x1)
                        x2c = min(self.last_depth.shape[1], x2)
                        crop = self.last_depth[y1c:y2c, x1c:x2c]
                        if crop.size > 0:
                            valid = crop[(crop > 0) & np.isfinite(crop)]
                            if valid.size > 0:
                                z = float(np.median(valid))

                    xyz = None
                    if z is not None and z > 0.0:
                        w = self.get_parameter('color_width').value
                        h = self.get_parameter('color_height').value
                        fov = math.radians(self.get_parameter('color_fov_deg').value)
                        fx = (w / 2.0) / math.tan(fov / 2.0)
                        fy = fx
                        X = (cx - w / 2.0) * z / fx
                        Y = (cy - h / 2.0) * z / fy
                        Z = z
                        xyz = (float(X), float(Y), float(Z))

                    detections.append({
                        'class': class_name,
                        'class_id': int(CLASS_TO_OBJ_ID[class_name]),
                        'conf': conf,
                        'bbox': (x1, y1, x2, y2),
                        'xyz': xyz,
                    })

        # simple policy: publish best detection
        if detections:
            best = max(detections, key=lambda d: d['conf'])
            msg = Int32()
            msg.data = int(best['class_id'])
            self.detect_pub.publish(msg)
            self.get_logger().info(f"Detected {best['class']} conf={best['conf']:.2f} xyz={best['xyz']}")

        # debug image
        debug = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(debug, f"{d['class']} {d['conf']:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
        try:
            dbg_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(debug, cv2.COLOR_RGB2BGR), encoding='bgr8')
            self.debug_pub.publish(dbg_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import time
import math

# Маппинг классов YOLO -> id миссии (поменяй при необходимости)
CLASS_TO_OBJ_ID = {
    "backpack": 24,
    "chair": 56,
    "bottle": 39,
    "laptop": 63,
    "cup": 41,
}

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()
        # модель: можно поменять на 'yolo26s.pt' для лучшей точности
        self.model = YOLO("yolo26n.pt")
        self.model_conf = 0.35
        self.model_iou = 0.45
        self.device = 'cpu'  # 'cuda' если доступна

        # подписки (bridge публикует эти топики)
        self.create_subscription(Image, '/aliengo/camera/color/image_raw', self.rgb_cb, 5)
        self.create_subscription(Image, '/aliengo/camera/depth/image_raw', self.depth_cb, 5)

        # публикуем найденный id как Int32 -> bridge ожидает '/aliengo/detected_object_id'
        self.detect_pub = self.create_publisher(Int32, '/aliengo/detected_object_id', 10)

        # (опционально) debug image
        self.debug_pub = self.create_publisher(Image, '/aliengo/detections/image', 1)

        self.last_depth = None
        self.depth_ts = 0.0

        # Параметры камеры (по умолчанию совпадают с `isaac_controller.py`), можно менять через ros2 params
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 360)
        self.declare_parameter("color_fov_deg", 70.0)

        self.get_logger().info("YOLO26 Object Detector запущен")

    def depth_cb(self, msg: Image):
        if msg.encoding in ("32FC1",):
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        else:
            # если пришёл uint16 или другой формат, попробуй конвертировать
            depth = np.frombuffer(msg.data, dtype=np.uint8)
            try:
                depth = depth.reshape((msg.height, msg.width))
                depth = depth.astype(np.float32)
            except Exception:
                return
        self.last_depth = depth
        self.depth_ts = time.time()

    def rgb_cb(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"CVBridge rgb error: {e}")
            return

        # подготавливаем изображение для модели (BGR ожидается OpenCV)
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # inference
        results = self.model.predict(
            source=img_bgr,
            conf=self.model_conf,
            iou=self.model_iou,
            device=self.device,
            verbose=False,
            stream=False,
        )

        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            # boxes.xyxy, boxes.cls, boxes.conf
            for i in range(len(boxes)):
                try:
                    xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else np.array(boxes.xyxy[i])
                    cls_id = int(boxes.cls[i]) if hasattr(boxes, "cls") else int(boxes.data[i,5])
                    conf = float(boxes.conf[i]) if hasattr(boxes, "conf") else float(boxes.data[i,4])
                    class_name = r.names[cls_id]
                except Exception:
                    continue

                if class_name not in CLASS_TO_OBJ_ID:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[:4])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # depth -> 3D (камера в своей системе координат)
                z = None
                if self.last_depth is not None and time.time() - self.depth_ts < 0.5:
                    crop = self.last_depth[max(0,y1):min(self.last_depth.shape[0],y2),
                                           max(0,x1):min(self.last_depth.shape[1],x2)]
                    if crop.size > 0:
                        valid = crop[crop > 0]
                        if valid.size > 0:
                            z = float(np.median(valid))
                # если depth нет — оставляем None

                # вычислим 3D точку (приближённо) при наличии z
                xyz = None
                if z is not None and z > 0.0:
                    w = self.get_parameter("color_width").value
                    h = self.get_parameter("color_height").value
                    fov = math.radians(self.get_parameter("color_fov_deg").value)
                    fx = (w / 2.0) / math.tan(fov / 2.0)
                    fy = fx  # приближение
                    X = (cx - w / 2.0) * z / fx
                    Y = (cy - h / 2.0) * z / fy
                    Z = z
                    xyz = (float(X), float(Y), float(Z))

                detections.append({
                    "class": class_name,
                    "class_id": int(CLASS_TO_OBJ_ID[class_name]),
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "xyz": xyz,
                })

        # simple policy: если есть детекция нужного класса — публикуем первый найденный id
        if detections:
            best = max(detections, key=lambda d: d["conf"])
            msg = Int32()
            msg.data = int(best["class_id"])
            self.detect_pub.publish(msg)
            self.get_logger().info(f"Detected {best['class']} conf={best['conf']:.2f} xyz={best['xyz']}")

        # publish debug image with boxes
        debug = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            cv2.rectangle(debug, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(debug, f"{d['class']} {d['conf']:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        try:
            dbg_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(debug, cv2.COLOR_RGB2BGR), encoding='bgr8')
            self.debug_pub.publish(dbg_msg)
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()