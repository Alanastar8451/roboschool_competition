"""
ROS2 Mission Node — finite state machine that navigates the robot
to each target object in sequence using a reference-image detector
and depth-based approach.
"""

import json
import math
import os
from enum import Enum, auto

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32

from ros2_bridge_pkg.reference_detector import (
    ReferenceImageDetectorBackend,
    Detection,
)


class State(Enum):
    WAIT_MISSION = auto()
    SEARCH = auto()
    ALIGN = auto()
    APPROACH = auto()
    REPORT = auto()
    WAIT_NEXT_TARGET = auto()
    FINISHED = auto()


# Camera geometry for RGB → depth coordinate mapping.
# RGB: 640×360, FOV_h=70°;  Depth: 848×480, FOV_h=86°.
RGB_W, RGB_H = 640, 360
DEPTH_W, DEPTH_H = 848, 480
RGB_CX, RGB_CY = RGB_W / 2.0, RGB_H / 2.0
DEPTH_CX, DEPTH_CY = DEPTH_W / 2.0, DEPTH_H / 2.0
RGB_FX = RGB_CX / math.tan(math.radians(70.0 / 2.0))
DEPTH_FX = DEPTH_CX / math.tan(math.radians(86.0 / 2.0))


def rgb_pixel_to_depth_pixel(x_rgb: float, y_rgb: float):
    """Map a point from the RGB image plane to the depth image plane."""
    x_depth = (x_rgb - RGB_CX) * (DEPTH_FX / RGB_FX) + DEPTH_CX
    y_depth = (y_rgb - RGB_CY) * (DEPTH_FX / RGB_FX) + DEPTH_CY
    return x_depth, y_depth


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")

        # ---- ROS parameters (overridable via CLI / YAML) ----
        self.declare_parameter("refs_dir", "")
        self.declare_parameter("debug", False)
        self.declare_parameter("search_wz", 0.6)
        self.declare_parameter("search_spiral_vx", 0.15)
        self.declare_parameter("search_advance_vx", 1.0)
        self.declare_parameter("search_scan_time", 11.0)
        self.declare_parameter("search_advance_time", 10.0)
        self.declare_parameter("obstacle_threshold", 2)
        self.declare_parameter("avoid_turn_time", 3)
        self.declare_parameter("align_pixel_tolerance", 40.0)
        self.declare_parameter("align_kp", 0.005)
        self.declare_parameter("approach_distance_threshold", 0.4)
        self.declare_parameter("approach_max_vx", 1.0)
        self.declare_parameter("approach_kp_dist", 0.4)
        self.declare_parameter("max_vx", 1.5)
        self.declare_parameter("max_wz", 1.5)
        self.declare_parameter("min_detection_score", 0.12)
        self.declare_parameter("lost_timeout_s", 3.0)

        self.refs_dir = self.get_parameter("refs_dir").value
        self.debug = self.get_parameter("debug").value
        self.search_wz = self.get_parameter("search_wz").value
        self.search_spiral_vx = self.get_parameter("search_spiral_vx").value
        self.search_advance_vx = self.get_parameter("search_advance_vx").value
        self.search_scan_time = self.get_parameter("search_scan_time").value
        self.search_advance_time = self.get_parameter("search_advance_time").value
        self.obstacle_threshold = self.get_parameter("obstacle_threshold").value
        self.avoid_turn_time = self.get_parameter("avoid_turn_time").value
        self.align_tol = self.get_parameter("align_pixel_tolerance").value
        self.align_kp = self.get_parameter("align_kp").value
        self.approach_dist = self.get_parameter("approach_distance_threshold").value
        self.approach_max_vx = self.get_parameter("approach_max_vx").value
        self.approach_kp_dist = self.get_parameter("approach_kp_dist").value
        self.max_vx = self.get_parameter("max_vx").value
        self.max_wz = self.get_parameter("max_wz").value
        self.min_score = self.get_parameter("min_detection_score").value
        self.lost_timeout = self.get_parameter("lost_timeout_s").value

        if not self.refs_dir:
            for candidate in [
                "/workspace/aliengo_competition/resources/assets/objects",
                os.path.expanduser(
                    "~/workspace/roboschool_competition/resources/assets/objects"
                ),
            ]:
                if os.path.isdir(candidate):
                    self.refs_dir = candidate
                    break

        self.get_logger().info(f"refs_dir = {self.refs_dir}")
        self.get_logger().info(f"debug    = {self.debug}")

        # ---- Detector ----
        self.detector = ReferenceImageDetectorBackend(
            refs_dir=self.refs_dir,
            min_matches=10,
            min_inliers=8,
            ransac_reproj_threshold=4.0,
            min_bbox_area=600,
            debug_visualization=self.debug,
        )

        # ---- Publishers ----
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.det_pub = self.create_publisher(
            Int32, "/aliengo/detected_object_id", 10
        )

        # ---- Subscribers ----
        self.create_subscription(
            Image, "/aliengo/camera/color/image_raw", self._rgb_cb, 10
        )
        self.create_subscription(
            Image, "/aliengo/camera/depth/image_raw", self._depth_cb, 10
        )
        self.create_subscription(
            TwistStamped, "/aliengo/base_velocity", self._vel_cb, 10
        )
        self.create_subscription(
            String, "/aliengo/mission_queue", self._queue_cb, 10
        )
        self.create_subscription(
            Int32, "/aliengo/current_target_id", self._target_cb, 10
        )
        self.create_subscription(
            String, "/aliengo/mission_status", self._status_cb, 10
        )

        # ---- Cached sensor data ----
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_vx = 0.0

        # ---- Mission state ----
        self.mission_queue: list | None = None
        self.sim_target_id: int = -1
        self.mission_status: dict = {}
        self.id_to_name: dict = {}

        # ---- FSM ----
        self.state = State.WAIT_MISSION
        self.last_detection: Detection | None = None
        self.last_detection_time = 0.0
        self.report_sent_id: int | None = None
        self.report_time = 0.0

        # ---- Search sub-phases ----
        self._search_phase = "SCAN"
        self._search_phase_start = 0.0
        self._search_cycle = 0
        self._search_direction = 1  # +1 CCW, -1 CW; flips each cycle
        self._avoid_direction = 1.0

        self._cb_counts = {
            "rgb": 0, "depth": 0, "vel": 0,
            "queue": 0, "target": 0, "status": 0,
        }
        self._wait_ticks = 0

        if self.debug:
            os.makedirs("debug_mission", exist_ok=True)
            self._dbg_frame = 0

        self.create_timer(0.1, self._tick)
        self.get_logger().info("Mission node started.")

    # ==================================================================
    # Callbacks — store latest data, no heavy processing
    # ==================================================================
    def _rgb_cb(self, msg: Image):
        self._cb_counts["rgb"] += 1
        try:
            self.latest_rgb = np.frombuffer(
                msg.data, dtype=np.uint8
            ).reshape((msg.height, msg.width, 3))
        except ValueError:
            pass

    def _depth_cb(self, msg: Image):
        self._cb_counts["depth"] += 1
        try:
            self.latest_depth = np.frombuffer(
                msg.data, dtype=np.float32
            ).reshape((msg.height, msg.width))
        except ValueError:
            pass

    def _vel_cb(self, msg: TwistStamped):
        self._cb_counts["vel"] += 1
        self.latest_vx = msg.twist.linear.x

    def _queue_cb(self, msg: String):
        self._cb_counts["queue"] += 1
        if self._cb_counts["queue"] == 1:
            self.get_logger().info(f"[CB] First mission_queue msg: {msg.data[:120]}")
        try:
            parsed = json.loads(msg.data)
            if isinstance(parsed, list) and len(parsed) > 0:
                self.mission_queue = parsed
                self.id_to_name = {int(item[0]): str(item[1]) for item in parsed}
        except (json.JSONDecodeError, IndexError) as e:
            self.get_logger().warn(f"[CB] queue parse error: {e}")

    def _target_cb(self, msg: Int32):
        self._cb_counts["target"] += 1
        if self._cb_counts["target"] == 1:
            self.get_logger().info(f"[CB] First current_target_id msg: {msg.data}")
        self.sim_target_id = msg.data

    def _status_cb(self, msg: String):
        self._cb_counts["status"] += 1
        try:
            self.mission_status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    # ==================================================================
    # Command helpers
    # ==================================================================
    def _send_cmd(self, vx: float, vy: float, wz: float):
        vx = max(-self.max_vx, min(self.max_vx, vx))
        vy = max(-self.max_vx, min(self.max_vx, vy))
        wz = max(-self.max_wz, min(self.max_wz, wz))
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._send_cmd(0.0, 0.0, 0.0)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ==================================================================
    # Obstacle helpers
    # ==================================================================
    def _forward_depth(self) -> float | None:
        """Median depth in the center strip of the depth image."""
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        cy1, cy2 = int(h * 0.35), int(h * 0.65)
        cx1, cx2 = int(w * 0.4), int(w * 0.6)
        roi = self.latest_depth[cy1:cy2, cx1:cx2]
        roi = np.abs(roi)
        valid = roi[(roi > 0.05) & (roi < 20.0) & np.isfinite(roi)]
        if valid.size < 4:
            return None
        return float(np.median(valid))

    def _forward_is_blocked(self, threshold: float | None = None) -> bool:
        """True when an obstacle is within *threshold* metres ahead."""
        if threshold is None:
            threshold = self.obstacle_threshold
        d = self._forward_depth()
        if d is None:
            return False
        return d < threshold

    # ==================================================================
    # Detection helpers
    # ==================================================================
    def _detect_target(self) -> Detection | None:
        if self.latest_rgb is None:
            return None
        target_name = self.id_to_name.get(self.sim_target_id)
        if target_name is None:
            return None

        dets = self.detector.detect(self.latest_rgb, target_name)
        if not dets:
            return None

        best = max(dets, key=lambda d: d.score)
        if best.score < self.min_score:
            return None

        if self.debug and self.latest_rgb is not None:
            vis = self.detector.draw_debug(self.latest_rgb, [best])
            self._dbg_frame += 1
            if self._dbg_frame % 10 == 0:
                path = f"debug_mission/frame_{self._dbg_frame:06d}.jpg"
                cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return best

    def _estimate_depth(self, det: Detection) -> float | None:
        """Median depth inside the central 50 % of the bbox mapped to depth image."""
        if self.latest_depth is None:
            return None

        x1, y1, x2, y2 = det.bbox_xyxy
        cx_rgb = (x1 + x2) / 2.0
        cy_rgb = (y1 + y2) / 2.0
        hw = (x2 - x1) * 0.25
        hh = (y2 - y1) * 0.25

        dx1, dy1 = rgb_pixel_to_depth_pixel(cx_rgb - hw, cy_rgb - hh)
        dx2, dy2 = rgb_pixel_to_depth_pixel(cx_rgb + hw, cy_rgb + hh)

        dh, dw = self.latest_depth.shape
        ix1 = max(0, int(dx1))
        iy1 = max(0, int(dy1))
        ix2 = min(dw, int(dx2))
        iy2 = min(dh, int(dy2))

        if ix2 <= ix1 or iy2 <= iy1:
            return None

        roi = self.latest_depth[iy1:iy2, ix1:ix2]
        roi = np.abs(roi)
        valid = roi[(roi > 0.05) & (roi < 20.0) & np.isfinite(roi)]

        if valid.size < 4:
            return None

        return float(np.median(valid))

    # ==================================================================
    # FSM tick
    # ==================================================================
    def _tick(self):
        handler = {
            State.WAIT_MISSION: self._tick_wait,
            State.SEARCH: self._tick_search,
            State.ALIGN: self._tick_align,
            State.APPROACH: self._tick_approach,
            State.REPORT: self._tick_report,
            State.WAIT_NEXT_TARGET: self._tick_wait_next,
            State.FINISHED: self._tick_finished,
        }
        handler[self.state]()

    # ---- WAIT_MISSION ----
    def _tick_wait(self):
        self._stop()
        self._wait_ticks += 1
        if self._wait_ticks % 30 == 1:
            self.get_logger().info(
                f"[WAIT] tick={self._wait_ticks} "
                f"queue={'set' if self.mission_queue else 'None'} "
                f"target_id={self.sim_target_id} "
                f"cbs={self._cb_counts}"
            )
        if self.mission_queue is not None and self.sim_target_id >= 0:
            self.get_logger().info(
                f"Mission received. Queue={self.mission_queue}, "
                f"first target id={self.sim_target_id}"
            )
            self._transition(State.SEARCH)

    # ---- SEARCH ----
    def _tick_search(self):
        det = self._detect_target()
        if det is not None:
            self.last_detection = det
            self.last_detection_time = self._now()
            target_name = self.id_to_name.get(self.sim_target_id, "?")
            self.get_logger().info(
                f"[SEARCH] Detected {target_name} "
                f"score={det.score:.3f} bbox={det.bbox_xyxy}"
            )
            self._transition(State.ALIGN)
            return

        now = self._now()
        elapsed = now - self._search_phase_start

        if self._search_phase == "SCAN":
            wz = self.search_wz * self._search_direction
            self._send_cmd(self.search_spiral_vx, 0.0, wz)
            if elapsed > self.search_scan_time:
                self._search_phase = "ADVANCE"
                self._search_phase_start = now
                self._search_cycle += 1
                self._search_direction *= -1
                self.get_logger().info(
                    f"[SEARCH] Scan done, advancing "
                    f"(cycle {self._search_cycle}, "
                    f"next_dir={'CCW' if self._search_direction > 0 else 'CW'})"
                )

        elif self._search_phase == "ADVANCE":
            if self._forward_is_blocked():
                self._search_phase = "AVOID"
                self._search_phase_start = now
                self._avoid_direction = (
                    1.0 if np.random.random() > 0.5 else -1.0
                )
                fwd = self._forward_depth()
                self.get_logger().info(
                    f"[SEARCH] Obstacle ahead ({fwd:.2f}m), avoiding"
                    if fwd else "[SEARCH] Obstacle ahead, avoiding"
                )
                return
            drift_wz = float(np.random.uniform(-0.15, 0.15))
            self._send_cmd(self.search_advance_vx, 0.0, drift_wz)
            if elapsed > self.search_advance_time:
                self._search_phase = "SCAN"
                self._search_phase_start = now
                self.get_logger().info("[SEARCH] Advance done, scanning again")

        elif self._search_phase == "AVOID":
            vy_dodge = 0.5 * self._avoid_direction
            self._send_cmd(-0.1, vy_dodge, self.search_wz * self._avoid_direction)
            if elapsed > self.avoid_turn_time or not self._forward_is_blocked():
                self._search_phase = "ADVANCE"
                self._search_phase_start = now

    # ---- ALIGN ----
    def _tick_align(self):
        det = self._detect_target()
        if det is not None:
            self.last_detection = det
            self.last_detection_time = self._now()
        elif self._now() - self.last_detection_time > self.lost_timeout:
            self.get_logger().warn("[ALIGN] Target lost, back to SEARCH")
            self._transition(State.SEARCH)
            return

        if self.last_detection is None:
            self._stop()
            return

        x1, _, x2, _ = self.last_detection.bbox_xyxy
        bbox_cx = (x1 + x2) / 2.0
        error_x = bbox_cx - RGB_CX

        if abs(error_x) < self.align_tol:
            self._transition(State.APPROACH)
            return

        wz = -self.align_kp * error_x

        dist = self._estimate_depth(self.last_detection)
        vx = 0.0
        if (dist is not None and dist > 4.0
                and abs(error_x) < self.align_tol * 3
                and not self._forward_is_blocked()):
            vx = 0.3

        self._send_cmd(vx, 0.0, wz)

    # ---- APPROACH ----
    def _tick_approach(self):
        det = self._detect_target()
        if det is not None:
            self.last_detection = det
            self.last_detection_time = self._now()
        elif self._now() - self.last_detection_time > self.lost_timeout:
            self.get_logger().warn("[APPROACH] Target lost, back to SEARCH")
            self._transition(State.SEARCH)
            return

        if self.last_detection is None:
            self._stop()
            return

        x1, _, x2, _ = self.last_detection.bbox_xyxy
        bbox_cx = (x1 + x2) / 2.0
        error_x = bbox_cx - RGB_CX
        wz = -self.align_kp * error_x

        dist = self._estimate_depth(self.last_detection)
        if dist is None:
            self._send_cmd(0.4, 0.0, wz)
            return

        if dist <= self.approach_dist:
            self._stop()
            self.get_logger().info(
                f"[APPROACH] Close enough: dist={dist:.2f}m"
            )
            self._transition(State.REPORT)
            return

        vx = min(self.approach_max_vx,
                 self.approach_kp_dist * (dist - self.approach_dist))
        vx = max(0.15, vx)

        self._send_cmd(vx, 0.0, wz)

    # ---- REPORT ----
    def _tick_report(self):
        self._stop()
        target_id = self.sim_target_id
        target_name = self.id_to_name.get(target_id, "?")

        msg = Int32()
        msg.data = target_id
        self.det_pub.publish(msg)

        self.report_sent_id = target_id
        self.report_time = self._now()
        self.get_logger().info(
            f"[REPORT] Reported object_id={target_id} ({target_name})"
        )
        self._transition(State.WAIT_NEXT_TARGET)

    # ---- WAIT_NEXT_TARGET ----
    def _tick_wait_next(self):
        self._stop()

        if self.mission_status.get("finished", False):
            self._transition(State.FINISHED)
            return

        if self.sim_target_id != self.report_sent_id:
            new_name = self.id_to_name.get(self.sim_target_id, "?")
            self.get_logger().info(
                f"[WAIT] Target changed to id={self.sim_target_id} ({new_name})"
            )
            self.last_detection = None
            self._transition(State.SEARCH)
            return

        if self._now() - self.report_time > 5.0:
            self.get_logger().warn("[WAIT] Timeout, re-publishing report")
            msg = Int32()
            msg.data = self.report_sent_id
            self.det_pub.publish(msg)
            self.report_time = self._now()

    # ---- FINISHED ----
    def _tick_finished(self):
        self._stop()

    # ---- transition helper ----
    def _transition(self, new_state: State):
        self.get_logger().info(f"FSM: {self.state.name} -> {new_state.name}")
        if new_state == State.SEARCH:
            self._search_phase = "SCAN"
            self._search_phase_start = self._now()
        self.state = new_state


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
