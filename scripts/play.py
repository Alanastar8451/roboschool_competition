from pathlib import Path
import sys
import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym

assert isaacgym
import torch
import numpy as np

import pickle as pkl
from pathlib import Path

from aliengo_gym.envs import *
from aliengo_gym.envs.base.legged_robot_config import Cfg
from aliengo_gym.envs.aliengo.aliengo_config import config_aliengo
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm


def resolve_latest_run_dir():
    runs_root = Path(__file__).resolve().parents[1] / "runs" / "gait-conditioned-agility"
    candidates = [path for path in runs_root.glob("*/train/*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No training runs found under {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)

def _load_policy_from_jit(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit', map_location='cpu')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit', map_location='cpu')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def _load_policy_from_state_dict(logdir, env):
    from aliengo_gym_learn.ppo_cse.actor_critic import ActorCritic

    actor_critic = ActorCritic(
        env.num_obs,
        env.num_privileged_obs,
        env.num_obs_history,
        env.num_actions,
    ).to('cpu')
    state_dict = torch.load(logdir + '/checkpoints/ac_weights_last.pt', map_location='cpu')
    actor_critic.load_state_dict(state_dict=state_dict, strict=True)
    actor_critic.eval()

    def policy(obs, info={}):
        with torch.no_grad():
            action = actor_critic.act_student(obs["obs_history"].to('cpu'), policy_info=info)
        return action

    return policy


def load_policy(logdir, env):
    # On CPU-only setups, JIT modules can crash with low-level c10 errors.
    # Prefer state_dict policy in that case.
    if not torch.cuda.is_available():
        print("CUDA is unavailable, loading policy from ac_weights_last.pt")
        return _load_policy_from_state_dict(logdir, env)
    try:
        return _load_policy_from_jit(logdir)
    except Exception as exc:
        print(f"Failed to load JIT policy ({exc}), falling back to ac_weights_last.pt")
        return _load_policy_from_state_dict(logdir, env)


def _load_pickle_with_cpu_torch(file_obj):
    # `parameters.pkl` can contain CUDA tensors. Force deserialization to CPU
    # so controller can start even when CUDA is unavailable.
    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    try:
        return pkl.load(file_obj)
    finally:
        torch.storage._load_from_bytes = original_loader


def load_env(logdir: Path, headless=False):
    with open(logdir / "parameters.pkl", 'rb') as file:
        pkl_cfg = _load_pickle_with_cpu_torch(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)
        from aliengo_gym_learn.ppo_cse.actor_critic import AC_Args
        ac_cfg = pkl_cfg.get("AC_Args", {})
        for key, value in ac_cfg.items():
            if hasattr(AC_Args, key):
                setattr(AC_Args, key, value)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.env.front_camera_enabled = True
    Cfg.env.front_camera_attach_body_name = "base"
    Cfg.env.front_camera_offset_xyz = [0.38, 0.0, 0.18]
    Cfg.env.front_camera_pitch_deg = 0.0
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.curriculum = False
    Cfg.terrain.measure_heights = False
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False

    Cfg.domain_rand.lag_timesteps = 0
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"

    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper

    sim_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if sim_device == 'cpu':
        Cfg.sim.use_gpu_pipeline = False
    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger

    policy = load_policy(str(logdir), env)

    return env, policy


def play_aliengo(headless=True):
    from ml_logger import logger

    from aliengo_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    logdir = resolve_latest_run_dir()
    env, policy = load_env(logdir, headless=headless)

    num_eval_steps = 1000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5],
             "galloping": [0.25, 0, 0]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25
    stance_length_cmd = 0.45

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        target_x_vels[i] = x_vel_cmd
        with torch.no_grad():
            actions = policy(obs)
        x_vel_cmd += 0.0015
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        if x_vel_cmd < 0.1:
            env.commands[:, 8] = 1.0
        else:
            env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        env.commands[:, 13] = stance_length_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = logdir / "play_diagnostics.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Matplotlib backend '{backend}' has no GUI; saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_aliengo(headless=False)
