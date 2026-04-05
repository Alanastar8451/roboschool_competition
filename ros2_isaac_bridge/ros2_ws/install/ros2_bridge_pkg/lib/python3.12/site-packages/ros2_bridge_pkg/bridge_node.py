import socket
import json

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image


CMD_IP = "127.0.0.1"
CMD_PORT = 5005

STATE_IP = "127.0.0.1"
STATE_PORT = 5006

RGB_IP = "127.0.0.1"
RGB_PORT = 5007

DEPTH_IP = "127.0.0.1"
DEPTH_PORT = 5008


class BridgeNode(Node):
    def __init__(self):
        super().__init__("bridge_node")

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind((STATE_IP, STATE_PORT))
        self.state_sock.setblocking(False)

        self.rgb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rgb_sock.bind((RGB_IP, RGB_PORT))
        self.rgb_sock.setblocking(False)

        self.depth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.depth_sock.bind((DEPTH_IP, DEPTH_PORT))
        self.depth_sock.setblocking(False)

        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_callback,
            10,
        )

        self.vel_pub = self.create_publisher(
            TwistStamped,
            "/aliengo/base_velocity",
            10,
        )

        self.rgb_pub = self.create_publisher(
            Image,
            "/aliengo/camera/color/image_raw",
            10,
        )

        self.depth_pub = self.create_publisher(
            Image,
            "/aliengo/camera/depth/image_raw",
            10,
        )

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("ROS bridge node started.")

    def cmd_callback(self, msg: Twist):
        payload = {
            "vx": msg.linear.x,
            "vy": msg.linear.y,
            "wz": msg.angular.z,
        }

        data = json.dumps(payload).encode("utf-8")
        self.cmd_sock.sendto(data, (CMD_IP, CMD_PORT))

    def timer_callback(self):
        try:
            data, _ = self.state_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))

            out = TwistStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = "base"

            out.twist.linear.x = float(msg.get("vx", 0.0))
            out.twist.linear.y = float(msg.get("vy", 0.0))
            out.twist.angular.z = float(msg.get("wz", 0.0))

            self.vel_pub.publish(out)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"state receive error: {e}")

        try:
            data, _ = self.rgb_sock.recvfrom(65535)

            np_arr = np.frombuffer(data, dtype=np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image_bgr is not None:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "front_camera"
                msg.height = image_rgb.shape[0]
                msg.width = image_rgb.shape[1]
                msg.encoding = "rgb8"
                msg.is_bigendian = 0
                msg.step = image_rgb.shape[1] * 3
                msg.data = image_rgb.tobytes()

                self.rgb_pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"rgb receive error: {e}")

        try:
            data, _ = self.depth_sock.recvfrom(65535)

            np_arr = np.frombuffer(data, dtype=np.uint8)
            depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth_image is not None:
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "front_camera_depth"
                msg.height = depth_image.shape[0]
                msg.width = depth_image.shape[1]
                msg.encoding = "32FC1"
                msg.is_bigendian = 0
                msg.step = depth_image.shape[1] * 4
                msg.data = depth_image.astype(np.float32).tobytes()

                self.depth_pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"depth receive error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()