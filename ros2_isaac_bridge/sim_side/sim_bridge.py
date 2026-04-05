import socket
import json
import time


CMD_IP = "127.0.0.1"
CMD_PORT = 5005

STATE_IP = "127.0.0.1"
STATE_PORT = 5006


class SimBridge:
    def __init__(self):
        self.latest_cmd = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }

        self.measured_state = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }

        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind((CMD_IP, CMD_PORT))
        self.cmd_sock.setblocking(False)

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    def update_controller(self):
        alpha = 0.1

        self.measured_state["vx"] += alpha * (self.latest_cmd["vx"] - self.measured_state["vx"])
        self.measured_state["vy"] += alpha * (self.latest_cmd["vy"] - self.measured_state["vy"])
        self.measured_state["wz"] += alpha * (self.latest_cmd["wz"] - self.measured_state["wz"])

    def send_state(self):
        msg = {
            "vx": self.measured_state["vx"],
            "vy": self.measured_state["vy"],
            "wz": self.measured_state["wz"],
            "timestamp": time.time(),
        }
        data = json.dumps(msg).encode("utf-8")
        self.state_sock.sendto(data, (STATE_IP, STATE_PORT))


def main():
    bridge = SimBridge()

    print("Sim bridge started.")
    print(f"Listening for commands on UDP {CMD_IP}:{CMD_PORT}")
    print(f"Sending state on UDP {STATE_IP}:{STATE_PORT}")

    while True:
        bridge.receive_cmd()
        bridge.update_controller()
        bridge.send_state()

        print(f"cmd={bridge.latest_cmd}  state={bridge.measured_state}")
        time.sleep(0.05)


if __name__ == "__main__":
    main()