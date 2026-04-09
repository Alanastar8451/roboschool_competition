#!/usr/bin/env python3
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image


class Controller(Node):
    def __init__(self):
        super().__init__("controller")

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

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = None
        self.last_log_time = 0.0
        self.log_interval = 1.0

        self.latest_vx = 0.0
        self.latest_vy = 0.0
        self.latest_wz_vel = 0.0
        self.latest_wz_imu = 0.0

        self.cached_frontiers = []
        self.cached_frontier_goal = None
        self.last_frontier_update_time = 0.0
        self.frontier_update_interval = 0.3

        self.depth_image = None

        self.UNKNOWN = -1
        self.FREE = 0
        self.OCCUPIED = 1

        self.resolution = 0.1

        # ===== LOCAL GRID =====
        self.grid_size = 100
        self.grid = np.full((self.grid_size, self.grid_size), self.UNKNOWN, dtype=np.int8)

        self.robot_gx = 10
        self.robot_gy = self.grid_size // 2

        # ===== GLOBAL GRID =====
        self.global_resolution = 0.1
        self.global_size_x = 200
        self.global_size_y = 120

        self.global_grid = np.full(
            (self.global_size_x, self.global_size_y),
            self.UNKNOWN,
            dtype=np.int8
        )

        self.global_origin_gx = self.global_size_x // 2
        self.global_origin_gy = self.global_size_y // 2

        self.min_local_free_cells = 15
        self.min_local_known_cells = 40
        self.prev_goal_dist = None
        self.no_progress_steps = 0
        self.max_no_progress_steps = 15
        self.min_progress_delta = 0.08
        
        self.blocked_turn_dir = 0.0
        self.blocked_turn_steps = 0

        self.recovery_mode = None
        self.recovery_steps_remaining = 0
        self.recovery_turn_dir = 1.0

        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0

        self.stuck_counter = 0
        self.stuck_trigger_steps = 4

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info("Controller started.")

    def imu_callback(self, msg: Imu):
        self.latest_wz_imu = msg.angular_velocity.z

    def vel_callback(self, msg: TwistStamped):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        wz = msg.twist.angular.z

        self.latest_vx = vx
        self.latest_vy = vy
        self.latest_wz_vel = wz

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self.last_time is None:
            self.last_time = now_sec
            return

        dt = now_sec - self.last_time
        self.last_time = now_sec

        if dt <= 0.0 or dt > 0.2:
            return

        self.yaw += wz * dt

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        self.x += (vx * cos_yaw - vy * sin_yaw) * dt
        self.y += (vx * sin_yaw + vy * cos_yaw) * dt

    def depth_callback(self, msg: Image):
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape((msg.height, msg.width))

        self.depth_image = depth

    def publish_cmd(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = wz
        self.cmd_pub.publish(msg)

    def timer_callback(self):
        self.update_grid_from_depth()

        now = time.time()
        if now - self.last_frontier_update_time >= self.frontier_update_interval:
            self.cached_frontiers = self.detect_frontiers_global()
            self.cached_frontier_goal = self.choose_nearest_frontier(self.cached_frontiers)
            self.last_frontier_update_time = now

        frontiers = self.cached_frontiers
        num_frontiers = len(frontiers)
        frontier_goal = self.cached_frontier_goal

        front_blocked = self.is_front_blocked()

        local_occ = int(np.sum(self.grid == self.OCCUPIED))
        local_free = int(np.sum(self.grid == self.FREE))
        local_unknown = int(np.sum(self.grid == self.UNKNOWN))

        local_known = local_occ + local_free
        local_map_valid = (
            local_free >= self.min_local_free_cells and
            local_known >= self.min_local_known_cells
        )

        if not local_map_valid:
            frontier_goal = None

        vx_cmd, vy_cmd, wz_cmd, mode = self.compute_frontier_cmd(frontier_goal)
        self.last_cmd_vx = vx_cmd
        self.last_cmd_wz = wz_cmd

        if self.is_stuck():
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.publish_cmd(vx_cmd, vy_cmd, wz_cmd)

        global_occ = int(np.sum(self.global_grid == self.OCCUPIED))
        global_free = int(np.sum(self.global_grid == self.FREE))
        global_unknown = int(np.sum(self.global_grid == self.UNKNOWN))

        if frontier_goal is None:
            frontier_text = "goal=None"
        else:
            goal_forward, goal_lateral = self.frontier_goal_to_local_xy(frontier_goal)
            frontier_text = (
                f"goal=({frontier_goal[0]},{frontier_goal[1]}) "
                f"goal_local=({goal_forward:.2f},{goal_lateral:.2f})"
            )

        current_time = time.time()

        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            
            self.get_logger().info(
                f"{mode} | "
                f"front_blocked={front_blocked} | "
                f"recovery_mode={self.recovery_mode} "
                f"recovery_steps={self.recovery_steps_remaining} "
                f"recovery_turn_dir={self.recovery_turn_dir:.1f} | "
                f"stuck_counter={self.stuck_counter} | "
                f"cmd_vx={vx_cmd:.3f}, cmd_wz={wz_cmd:.3f} | "
                f"meas_vx={self.latest_vx:.3f}, meas_vy={self.latest_vy:.3f}, "
                f"wz_vel={self.latest_wz_vel:.3f}, wz_imu={self.latest_wz_imu:.3f} | "
                f"x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f}"
            )

            self.get_logger().info(
                f"local_occ={local_occ} local_free={local_free} local_unknown={local_unknown} | "
                f"global_occ={global_occ} global_free={global_free} global_unknown={global_unknown} | "
                f"frontiers={num_frontiers} {frontier_text}"
            )

    def mark_ray_free(self, gx0, gy0, gx1, gy1):
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)

        x = gx0
        y = gy0

        sx = 1 if gx1 >= gx0 else -1
        sy = 1 if gy1 >= gy0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != gx1:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if self.grid[x, y] != self.OCCUPIED:
                        self.grid[x, y] = self.FREE
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != gy1:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if self.grid[x, y] != self.OCCUPIED:
                        self.grid[x, y] = self.FREE
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

    def mark_ray_free_global(self, gx0, gy0, gx1, gy1):
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)

        x = gx0
        y = gy0

        sx = 1 if gx1 >= gx0 else -1
        sy = 1 if gy1 >= gy0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != gx1:
                if 0 <= x < self.global_size_x and 0 <= y < self.global_size_y:
                    if self.global_grid[x, y] != self.OCCUPIED:
                        self.global_grid[x, y] = self.FREE
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != gy1:
                if 0 <= x < self.global_size_x and 0 <= y < self.global_size_y:
                    if self.global_grid[x, y] != self.OCCUPIED:
                        self.global_grid[x, y] = self.FREE
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

    def update_grid_from_depth(self):
        if self.depth_image is None:
            return
        
        self.grid.fill(self.UNKNOWN)
        
        h, w = self.depth_image.shape

        fx = w / 2.0
        fy = h / 2.0
        cx = w / 2.0
        cy = h / 2.0

        for v in range(0, h, 8):
            for u in range(0, w, 8):
                z = float(self.depth_image[v, u])

                if not np.isfinite(z):
                    continue
                if z <= 0.15 or z > 4.0:
                    continue

                x_cam = (u - cx) * z / fx

                forward = z
                lateral = -x_cam

                gx = self.robot_gx + int(forward / self.resolution)
                gy = self.robot_gy + int(lateral / self.resolution)

                cos_yaw = math.cos(self.yaw)
                sin_yaw = math.sin(self.yaw)

                global_x = self.x + forward * cos_yaw - lateral * sin_yaw
                global_y = self.y + forward * sin_yaw + lateral * cos_yaw

                ggx = self.global_origin_gx + int(global_x / self.global_resolution)
                ggy = self.global_origin_gy + int(global_y / self.global_resolution)

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.mark_ray_free(self.robot_gx, self.robot_gy, gx, gy)
                    self.grid[gx, gy] = self.OCCUPIED

                robot_global_gx = self.global_origin_gx + int(self.x / self.global_resolution)
                robot_global_gy = self.global_origin_gy + int(self.y / self.global_resolution)

                if 0 <= ggx < self.global_size_x and 0 <= ggy < self.global_size_y:
                    self.mark_ray_free_global(robot_global_gx, robot_global_gy, ggx, ggy)
                    self.global_grid[ggx, ggy] = self.OCCUPIED

    def detect_frontiers(self):
        frontiers = []

        for gx in range(1, self.grid_size - 1):
            for gy in range(1, self.grid_size - 1):
                if self.grid[gx, gy] != self.FREE:
                    continue

                has_unknown_neighbor = False

                for nx in range(gx - 1, gx + 2):
                    for ny in range(gy - 1, gy + 2):
                        if nx == gx and ny == gy:
                            continue

                        if self.grid[nx, ny] == self.UNKNOWN:
                            has_unknown_neighbor = True
                            break

                    if has_unknown_neighbor:
                        break

                if has_unknown_neighbor:
                    frontiers.append((gx, gy))

        return frontiers
    
    def detect_frontiers_global(self):
        frontiers = []

        for gx in range(1, self.global_size_x - 1):
            for gy in range(1, self.global_size_y - 1):
                if self.global_grid[gx, gy] != self.FREE:
                    continue

                has_unknown = False

                for nx in range(gx - 1, gx + 2):
                    for ny in range(gy - 1, gy + 2):
                        if nx == gx and ny == gy:
                            continue

                        if self.global_grid[nx, ny] == self.UNKNOWN:
                            has_unknown = True
                            break

                    if has_unknown:
                        break

                if has_unknown:
                    frontiers.append((gx, gy))

        return frontiers

    def choose_nearest_frontier(self, frontiers):
        if len(frontiers) == 0:
            return None

        robot_global_gx = self.global_origin_gx + int(self.x / self.global_resolution)
        robot_global_gy = self.global_origin_gy + int(self.y / self.global_resolution)

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        best = None
        best_score = None

        for gx, gy in frontiers:
            dx_global = (gx - robot_global_gx) * self.global_resolution
            dy_global = (gy - robot_global_gy) * self.global_resolution

            forward = dx_global * cos_yaw + dy_global * sin_yaw
            lateral = -dx_global * sin_yaw + dy_global * cos_yaw
            dist = math.hypot(dx_global, dy_global)

            if forward < 0.5:
                continue
            if abs(lateral) > 1.2:
                continue
            if dist > 2.5:
                continue

            score = dist + 0.8 * abs(lateral)

            if best is None or score < best_score:
                best = (gx, gy)
                best_score = score

        return best
    
    def frontier_goal_to_local_xy(self, frontier_goal):
        if frontier_goal is None:
            return None, None

        gx, gy = frontier_goal

        robot_global_gx = self.global_origin_gx + int(self.x / self.global_resolution)
        robot_global_gy = self.global_origin_gy + int(self.y / self.global_resolution)

        dx_global = (gx - robot_global_gx) * self.global_resolution
        dy_global = (gy - robot_global_gy) * self.global_resolution

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        forward = dx_global * cos_yaw + dy_global * sin_yaw
        lateral = -dx_global * sin_yaw + dy_global * cos_yaw

        return forward, lateral
    
    def compute_frontier_cmd(self, frontier_goal):
        if self.recovery_mode == "BACKUP":
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.recovery_mode = "TURN"
                self.recovery_steps_remaining = 18
            return -0.20, 0.0, 0.0, "RECOVERY_BACKUP"

        if self.recovery_mode == "TURN":
            self.recovery_steps_remaining -= 1
            if self.recovery_steps_remaining <= 0:
                self.recovery_mode = None
            return 0.0, 0.0, 0.6 * self.recovery_turn_dir, "RECOVERY_TURN"

        if self.stuck_counter >= self.stuck_trigger_steps:
            self.recovery_turn_dir = 1.0 if self.latest_wz_imu <= 0.0 else -1.0
            self.recovery_mode = "BACKUP"
            self.recovery_steps_remaining = 15
            self.stuck_counter = 0
            return -0.20, 0.0, 0.0, "STUCK_BACKUP"

        if frontier_goal is None:
            return 0.0, 0.0, 0.25, "SEARCH"

        goal_forward, goal_lateral = self.frontier_goal_to_local_xy(frontier_goal)

        if goal_forward is None:
            return 0.0, 0.0, 0.25, "SEARCH"

        heading_error = math.atan2(goal_lateral, goal_forward)

        wz = 0.9 * heading_error
        wz = max(min(wz, 0.6), -0.6)

        front_blocked = self.is_front_blocked()
        if front_blocked:
            self.start_recovery(frontier_goal)
            return -0.20, 0.0, 0.0, "RECOVERY_BACKUP"

        if abs(heading_error) > 0.9:
            return 0.0, 0.0, wz, "TURN_TO_FRONTIER"

        vx = 0.22 * max(0.0, math.cos(heading_error))

        if abs(heading_error) > 0.45:
            vx *= 0.55

        if goal_forward < 0.45:
            return 0.0, 0.0, 0.3 if heading_error >= 0.0 else -0.3, "REORIENT_TO_FRONTIER"

        return vx, 0.0, wz, "GO_TO_FRONTIER"

    def is_front_blocked(self):
        forward_min = 0.2
        forward_max = 0.9
        lateral_limit = 0.40

        gx_min = self.robot_gx + int(forward_min / self.resolution)
        gx_max = self.robot_gx + int(forward_max / self.resolution)
        gy_min = self.robot_gy - int(lateral_limit / self.resolution)
        gy_max = self.robot_gy + int(lateral_limit / self.resolution)

        gx_min = max(0, gx_min)
        gx_max = min(self.grid_size - 1, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(self.grid_size - 1, gy_max)

        occupied_count = 0
        near_hit = False

        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                if self.grid[gx, gy] == self.OCCUPIED:
                    occupied_count += 1

                    if gx <= self.robot_gx + int(0.45 / self.resolution):
                        near_hit = True

        if near_hit:
            return True

        return occupied_count >= 4

    def choose_blocked_turn_direction(self, frontier_goal):
        if frontier_goal is None:
            return 1.0

        goal_forward, goal_lateral = self.frontier_goal_to_local_xy(frontier_goal)

        if goal_lateral > 0.05:
            return 1.0
        elif goal_lateral < -0.05:
            return -1.0
        else:
            if self.blocked_turn_dir != 0.0:
                return self.blocked_turn_dir
            return 1.0
        
    def start_recovery(self, frontier_goal):
        self.recovery_turn_dir = self.choose_blocked_turn_direction(frontier_goal)
        self.recovery_mode = "BACKUP"
        self.recovery_steps_remaining = 15

    def is_stuck(self):
        cmd_trying_forward = self.last_cmd_vx > 0.10
        moving_forward_poorly = abs(self.latest_vx) < 0.04

        cmd_trying_turn = abs(self.last_cmd_wz) > 0.45
        turning_poorly = abs(self.latest_wz_imu) < 0.05

        if cmd_trying_forward and moving_forward_poorly:
            return True

        if cmd_trying_turn and turning_poorly and self.last_cmd_vx < 0.05:
            return True

        return False


def main(args=None):
    rclpy.init(args=args)
    node = Controller()

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