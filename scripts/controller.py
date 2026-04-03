from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aliengo_competition.common.helpers import get_args
from aliengo_competition.controllers.main_controller import run
from aliengo_competition.robot_interface.factory import make_robot_interface


def controller(args):
    robot = make_robot_interface(
        args=args,
        task=args.task,
        mode=args.mode,
        headless=args.headless,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    run(
        robot,
        steps=args.steps,
        render_camera=args.render_camera,
        camera_depth_max_m=args.camera_depth_max_m,
    )


if __name__ == "__main__":
    controller(get_args())
