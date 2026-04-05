import time
from sim_bridge_client import SimBridgeClient


def main():
    bridge = SimBridgeClient()

    measured_state = {
        "vx": 0.0,
        "vy": 0.0,
        "wz": 0.0,
    }

    alpha = 0.1

    print("Fake controller started.")

    while True:
        cmd = bridge.receive_cmd()

        measured_state["vx"] += alpha * (cmd["vx"] - measured_state["vx"])
        measured_state["vy"] += alpha * (cmd["vy"] - measured_state["vy"])
        measured_state["wz"] += alpha * (cmd["wz"] - measured_state["wz"])

        bridge.send_state(
            vx=measured_state["vx"],
            vy=measured_state["vy"],
            wz=measured_state["wz"],
        )

        print(f"cmd={cmd}  state={measured_state}")
        time.sleep(0.05)


if __name__ == "__main__":
    main()