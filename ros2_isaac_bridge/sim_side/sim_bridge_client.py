import socket
import json
import time
import cv2


CMD_IP = "127.0.0.1"
CMD_PORT = 5005

STATE_IP = "127.0.0.1"
STATE_PORT = 5006

RGB_IP = "127.0.0.1"
RGB_PORT = 5007

DEPTH_IP = "127.0.0.1"
DEPTH_PORT = 5008


class SimBridgeClient:
    def __init__(self):
        self.latest_cmd = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind((CMD_IP, CMD_PORT))
        self.cmd_sock.setblocking(False)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rgb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.depth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def receive_cmd(self):
        try:
            data, _ = self.cmd_sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            self.latest_cmd["vx"] = float(msg.get("vx", 0.0))
            self.latest_cmd["vy"] = float(msg.get("vy", 0.0))
            self.latest_cmd["wz"] = float(msg.get("wz", 0.0))
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"receive_cmd error: {e}")

        return self.latest_cmd.copy()

    def send_state(self, vx, vy, wz):
        msg = {
            "vx": float(vx),
            "vy": float(vy),
            "wz": float(wz),
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.state_sock.sendto(data, (STATE_IP, STATE_PORT))

    def send_rgb(self, rgb):
        success, encoded = cv2.imencode(".jpg", rgb)
        if not success:
            print("send_rgb error: JPEG encoding failed")
            return

        data = encoded.tobytes()
        self.rgb_sock.sendto(data, (RGB_IP, RGB_PORT))

    def send_depth(self, depth):
        depth_clipped = depth.copy()
        depth_clipped = depth_clipped.astype("float32")

        success, encoded = cv2.imencode(".png", depth_clipped)
        if not success:
            print("send_depth error: PNG encoding failed")
            return

        data = encoded.tobytes()
        self.depth_sock.sendto(data, (DEPTH_IP, DEPTH_PORT))