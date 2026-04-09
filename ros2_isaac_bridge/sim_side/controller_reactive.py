#!/usr/bin/env python3
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Imu, Image


class ReactiveExplorer(Node):
    def __init__(self):
        super().__init__("controller_reactive")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.vel_sub = self.create_subscription(
            TwistStamped,
            "/aliengo/base_velocity",
            self.vel_callback,
            10,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            "/aliengo/imu",
            self.imu_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/aliengo/camera/depth/image_raw",
            self.depth_callback,
            10,
        )

        self.depth_image = None
        self.latest_vx = 0.0
        self.latest_vy = 0.0
        self.latest_wz_vel = 0.0
        self.latest_wz_imu = 0.0

        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0

        self.last_log_time = 0.0
        self.log_interval = 1.0

        self.preferred_turn_dir = 1.0   # +1 left, -1 right
        self.state = "SCAN"
        self.state_steps = 0
        self.stuck_counter = 0

        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("Reactive explorer started.")

    # -----------------------------
    # ROS callbacks
    # -----------------------------
    def imu_callback(self, msg: Imu):
        self.latest_wz_imu = msg.angular_velocity.z

    def vel_callback(self, msg: TwistStamped):
        self.latest_vx = msg.twist.linear.x
        self.latest_vy = msg.twist.linear.y
        self.latest_wz_vel = msg.twist.angular.z

    def depth_callback(self, msg: Image):
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape((msg.height, msg.width))
        self.depth_image = depth

    # -----------------------------
    # Utilities
    # -----------------------------
    def publish_cmd(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = wz
        self.cmd_pub.publish(msg)
        self.last_cmd_vx = vx
        self.last_cmd_wz = wz

    def clip(self, x, lo, hi):
        return max(lo, min(hi, x))

    def valid_depth_mask(self, arr):
        return np.isfinite(arr) & (arr > 0.15) & (arr < 4.0)

    # -----------------------------
    # Depth sector analysis
    # -----------------------------
    def get_sector_stats(self):
        if self.depth_image is None:
            return None

        depth = self.depth_image
        h, w = depth.shape

        # Lower-middle image: where near-floor obstacles are most useful.
        row0 = int(0.38 * h)
        row1 = int(0.88 * h)
        roi = depth[row0:row1, :]

        # 5 angular sectors.
        edges = [
            0,
            int(0.20 * w),
            int(0.40 * w),
            int(0.60 * w),
            int(0.80 * w),
            w,
        ]
        names = ["far_left", "left", "center", "right", "far_right"]

        stats = {}
        for i, name in enumerate(names):
            sector = roi[:, edges[i]:edges[i + 1]]
            valid = self.valid_depth_mask(sector)

            if np.count_nonzero(valid) == 0:
                min_depth = 0.0
                mean_depth = 0.0
                valid_ratio = 0.0
            else:
                vals = sector[valid]
                min_depth = float(np.min(vals))
                mean_depth = float(np.mean(vals))
                valid_ratio = float(np.mean(valid))

            stats[name] = {
                "min": min_depth,
                "mean": mean_depth,
                "valid_ratio": valid_ratio,
            }

        # Narrow emergency center window.
        c0 = int(0.42 * w)
        c1 = int(0.58 * w)
        r0 = int(0.48 * h)
        r1 = int(0.88 * h)
        emergency = depth[r0:r1, c0:c1]
        valid = self.valid_depth_mask(emergency)
        emergency_min = float(np.min(emergency[valid])) if np.count_nonzero(valid) > 0 else 0.0

        return stats, emergency_min

    def sector_score(self, sector):
        # High score = safer/more attractive.
        # Mean depth dominates, min depth protects against collisions,
        # valid ratio penalizes empty/noisy sectors.
        return 0.65 * sector["mean"] + 0.35 * sector["min"] + 0.5 * sector["valid_ratio"]

    def choose_turn_direction(self, stats):
        left_score = 0.7 * self.sector_score(stats["left"]) + 0.3 * self.sector_score(stats["far_left"])
        right_score = 0.7 * self.sector_score(stats["right"]) + 0.3 * self.sector_score(stats["far_right"])

        if abs(left_score - right_score) < 0.08:
            return self.preferred_turn_dir

        return 1.0 if left_score > right_score else -1.0

    # -----------------------------
    # State helpers
    # -----------------------------
    def enter_state(self, new_state, steps, turn_dir=None):
        self.state = new_state
        self.state_steps = steps
        if turn_dir is not None:
            self.preferred_turn_dir = turn_dir

    def is_stuck(self):
        trying_forward = self.last_cmd_vx > 0.10
        not_moving_forward = abs(self.latest_vx) < 0.03

        trying_turn = abs(self.last_cmd_wz) > 0.35
        not_turning = abs(self.latest_wz_imu) < 0.08

        if trying_forward and not_moving_forward:
            return True

        if trying_turn and not_turning and self.last_cmd_vx < 0.03:
            return True

        return False

    # -----------------------------
    # Main control
    # -----------------------------
    def timer_callback(self):
        info = self.get_sector_stats()
        if info is None:
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        stats, emergency_min = info

        center_min = stats["center"]["min"]
        center_mean = stats["center"]["mean"]
        left_min = stats["left"]["min"]
        right_min = stats["right"]["min"]
        left_score = self.sector_score(stats["left"]) + 0.4 * self.sector_score(stats["far_left"])
        right_score = self.sector_score(stats["right"]) + 0.4 * self.sector_score(stats["far_right"])

        hard_block = (emergency_min > 0.0 and emergency_min < 0.42)
        center_blocked = (center_min > 0.0 and center_min < 0.75)
        center_tight = (center_mean > 0.0 and center_mean < 1.10)
        left_blocked = (left_min > 0.0 and left_min < 0.55)
        right_blocked = (right_min > 0.0 and right_min < 0.55)

        turn_dir = self.choose_turn_direction(stats)

        if self.is_stuck():
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        if self.stuck_counter >= 4:
            self.enter_state("BACKUP", 10, turn_dir=-1.0 if self.preferred_turn_dir > 0 else 1.0)
            self.stuck_counter = 0

        mode = self.state
        vx_cmd = 0.0
        wz_cmd = 0.0

        if self.state == "BACKUP":
            vx_cmd = -0.18
            wz_cmd = 0.0
            self.state_steps -= 1
            mode = "BACKUP"
            if self.state_steps <= 0:
                self.enter_state("TURN", 14, turn_dir=self.preferred_turn_dir)

        elif self.state == "TURN":
            vx_cmd = 0.0
            wz_cmd = 0.55 * self.preferred_turn_dir
            self.state_steps -= 1
            mode = "TURN"
            if self.state_steps <= 0:
                self.enter_state("SCAN", 0, turn_dir=self.preferred_turn_dir)

        else:
            # Reactive navigation only.
            if hard_block:
                self.enter_state("BACKUP", 8, turn_dir=turn_dir)
                vx_cmd = -0.18
                wz_cmd = 0.0
                mode = "EMERGENCY_BACKUP"

            elif center_blocked:
                self.preferred_turn_dir = turn_dir
                vx_cmd = 0.0
                wz_cmd = 0.45 * turn_dir
                mode = "AVOID_TURN"

            elif left_blocked and not right_blocked:
                self.preferred_turn_dir = -1.0
                vx_cmd = 0.12
                wz_cmd = -0.28
                mode = "EDGE_RIGHT"

            elif right_blocked and not left_blocked:
                self.preferred_turn_dir = 1.0
                vx_cmd = 0.12
                wz_cmd = 0.28
                mode = "EDGE_LEFT"

            else:
                # Open or semi-open region: keep moving with a mild bias.
                bias = self.clip(0.22 * (left_score - right_score), -0.28, 0.28)

                if center_tight:
                    vx_cmd = 0.10
                    wz_cmd = bias if abs(bias) > 0.06 else 0.18 * self.preferred_turn_dir
                    mode = "CAREFUL_FORWARD"
                else:
                    vx_cmd = 0.20
                    wz_cmd = 0.7 * bias
                    mode = "FORWARD"

                # If both sides are very open and center is good, keep heading stable.
                if center_mean > 1.8 and abs(left_score - right_score) < 0.12:
                    wz_cmd *= 0.4
                    mode = "FORWARD_STABLE"

        self.publish_cmd(vx_cmd, 0.0, wz_cmd)

        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            self.get_logger().info(
                f"{mode} | "
                f"state={self.state} steps={self.state_steps} pref_turn={self.preferred_turn_dir:+.1f} | "
                f"stuck_counter={self.stuck_counter} | "
                f"cmd_vx={vx_cmd:.3f} cmd_wz={wz_cmd:.3f} | "
                f"meas_vx={self.latest_vx:.3f} meas_vy={self.latest_vy:.3f} "
                f"wz_vel={self.latest_wz_vel:.3f} wz_imu={self.latest_wz_imu:.3f}"
            )
            self.get_logger().info(
                f"center[min={center_min:.2f}, mean={center_mean:.2f}] "
                f"left[min={left_min:.2f}, score={left_score:.2f}] "
                f"right[min={right_min:.2f}, score={right_score:.2f}] "
                f"emergency_min={emergency_min:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveExplorer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
