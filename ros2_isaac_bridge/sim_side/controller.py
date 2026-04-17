#!/usr/bin/env python3
import math
import random
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, Imu, JointState


class HLInterfaceController(Node):
    """
    Prototype controller for competitors.

    What it does:
    - publishes velocity commands to /cmd_vel
    - subscribes to robot state topics published by bridge_node
    - exposes high-level helper functions that can reused
    - sends random commands periodically as a demo
    """

    def __init__(self):
        super().__init__("controller")

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.vel_sub = self.create_subscription(
            TwistStamped,
            "/aliengo/base_velocity",
            self._vel_callback,
            10,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            "/aliengo/joint_states",
            self._joint_callback,
            10,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/aliengo/imu",
            self._imu_callback,
            10,
        )
        self.rgb_sub = self.create_subscription(
            Image,
            "/aliengo/camera/color/image_raw",
            self._rgb_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/aliengo/camera/depth/image_raw",
            self._depth_callback,
            10,
        )

        # ---------------- Cached state ----------------
        self.latest_base_velocity = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
            "stamp_sec": None,
        }

        self.latest_joint_state = {
            "names": [],
            "position": [],
            "velocity": [],
            "name_to_index": {},
            "stamp_sec": None,
        }

        self.latest_imu = {
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "stamp_sec": None,
        }

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_info = {
            "height": 0,
            "width": 0,
            "encoding": None,
            "stamp_sec": None,
        }

        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_info = {
            "height": 0,
            "width": 0,
            "encoding": None,
            "stamp_sec": None,
        }

        # ---------------- Demo behavior ----------------
        self.command_duration = 2.0
        self.last_command_change_time = self._now_sec()
        self.current_demo_cmd = {"vx": 2.0, "vy": 0.0, "wz": 0.0} #
        print("ros2 Hello")
        self.demo_enabled = True

        self.log_period = 1.0
        self.last_log_time = 0.0

        self.create_timer(0.05, self._main_loop)

        self.get_logger().info("Controller started.")
        self.get_logger().info("This node publishes random demo commands and exposes HL helper functions.")

    # =====================================================================
    # High-level API competitors can use
    # =====================================================================
    def send_command(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        self.send_command(0.0, 0.0, 0.0)

    def get_base_velocity(self) -> Dict[str, float]:
        return dict(self.latest_base_velocity)

    def get_vx(self) -> float:
        return float(self.latest_base_velocity["vx"])

    def get_vy(self) -> float:
        return float(self.latest_base_velocity["vy"])

    def get_wz(self) -> float:
        return float(self.latest_base_velocity["wz"])

    def get_joint_names(self) -> List[str]:
        return list(self.latest_joint_state["names"])

    def get_joint_positions(self) -> Dict[str, float]:
        names = self.latest_joint_state["names"]
        pos = self.latest_joint_state["position"]
        return {name: float(value) for name, value in zip(names, pos)}

    def get_joint_velocities(self) -> Dict[str, float]:
        names = self.latest_joint_state["names"]
        vel = self.latest_joint_state["velocity"]
        return {name: float(value) for name, value in zip(names, vel)}

    def get_joint_position(self, joint_name: str) -> Optional[float]:
        idx = self.latest_joint_state["name_to_index"].get(joint_name)
        if idx is None:
            return None
        return float(self.latest_joint_state["position"][idx])

    def get_joint_velocity(self, joint_name: str) -> Optional[float]:
        idx = self.latest_joint_state["name_to_index"].get(joint_name)
        if idx is None:
            return None
        return float(self.latest_joint_state["velocity"][idx])

    def get_imu(self) -> Dict[str, float]:
        return dict(self.latest_imu)

    def get_rgb_image(self) -> Optional[np.ndarray]:
        if self.latest_rgb is None:
            return None
        return self.latest_rgb.copy()

    def get_depth_image(self) -> Optional[np.ndarray]:
        if self.latest_depth is None:
            return None
        return self.latest_depth.copy()

    def get_depth_center(self) -> Optional[float]:
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        center_value = self.latest_depth[h // 2, w // 2]
        if not np.isfinite(center_value):
            return None
        return float(center_value)

    def robot_state_ready(self) -> bool:
        return (
            self.latest_base_velocity["stamp_sec"] is not None
            and self.latest_joint_state["stamp_sec"] is not None
            and self.latest_imu["stamp_sec"] is not None
        )

    # =====================================================================
    # Example competitor entry point
    # =====================================================================
    def run_user_code(self) -> None:
        """
        Competitors can replace the contents of this method with their own logic.
        """
        now = self._now_sec()

        if now - self.last_command_change_time >= self.command_duration:
            self.last_command_change_time = now
            self.current_demo_cmd = self._sample_random_command()
            self.get_logger().info(
                "New random command: "
                f"vx={self.current_demo_cmd['vx']:.3f}, "
                f"vy={self.current_demo_cmd['vy']:.3f}, "
                f"wz={self.current_demo_cmd['wz']:.3f}"
            )

        self.send_command(
            self.current_demo_cmd["vx"],
            self.current_demo_cmd["vy"],
            self.current_demo_cmd["wz"],
        )

    # =====================================================================
    # Internal callbacks
    # =====================================================================
    def _vel_callback(self, msg: TwistStamped) -> None:
        self.latest_base_velocity = {
            "vx": float(msg.twist.linear.x),
            "vy": float(msg.twist.linear.y),
            "wz": float(msg.twist.angular.z),
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _joint_callback(self, msg: JointState) -> None:
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        self.latest_joint_state = {
            "names": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "name_to_index": name_to_index,
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _imu_callback(self, msg: Imu) -> None:
        self.latest_imu = {
            "wx": float(msg.angular_velocity.x),
            "wy": float(msg.angular_velocity.y),
            "wz": float(msg.angular_velocity.z),
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _rgb_callback(self, msg: Image) -> None:
        try:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except ValueError:
            self.get_logger().warning("Failed to reshape RGB image.")
            return

        self.latest_rgb = image.copy()
        self.latest_rgb_info = {
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        except ValueError:
            self.get_logger().warning("Failed to reshape depth image.")
            return

        self.latest_depth = depth.copy()
        self.latest_depth_info = {
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "stamp_sec": self._msg_time_to_sec(msg.header.stamp),
        }

    # =====================================================================
    # Main loop
    # =====================================================================
    def _main_loop(self) -> None:
        if self.demo_enabled:
            self.run_user_code()

        now = self._now_sec()
        if now - self.last_log_time >= self.log_period:
            self.last_log_time = now
            self._log_status()

    def _log_status(self) -> None:
        vel = self.get_base_velocity()
        imu = self.get_imu()
        depth_center = self.get_depth_center()

        fl_hip = self.get_joint_position("FL_hip_joint")
        fr_hip = self.get_joint_position("FR_hip_joint")

        depth_text = "None" if depth_center is None else f"{depth_center:.3f}"
        fl_text = "None" if fl_hip is None else f"{fl_hip:.3f}"
        fr_text = "None" if fr_hip is None else f"{fr_hip:.3f}"

        self.get_logger().info(
            "state | "
            f"ready={self.robot_state_ready()} | "
            f"cmd=(vx={self.current_demo_cmd['vx']:.3f}, vy={self.current_demo_cmd['vy']:.3f}, wz={self.current_demo_cmd['wz']:.3f}) | "
            f"vel=(vx={vel['vx']:.3f}, vy={vel['vy']:.3f}, wz={vel['wz']:.3f}) | "
            f"imu_wz={imu['wz']:.3f} | "
            f"depth_center={depth_text} | "
            f"FL_hip={fl_text} | "
            f"FR_hip={fr_text}"
        )

    # =====================================================================
    # Utilities
    # =====================================================================
    def _sample_random_command(self) -> Dict[str, float]:
        vx = random.uniform(-0.8, 0.8)
        vy = random.uniform(-0.2, 0.2)
        wz = random.uniform(-0.8, 0.8)

        if abs(vx) < 0.08:
            vx = 0.0
        if abs(vy) < 0.05:
            vy = 0.0
        if abs(wz) < 0.08:
            wz = 0.0

        return {
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
        }

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _msg_time_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = HLInterfaceController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
