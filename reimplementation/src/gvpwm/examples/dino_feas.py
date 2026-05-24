# train feasibility with dino things
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import gym
import torch
import numpy as np
from omegaconf import OmegaConf


from ..adapters.dino_wm import DinoWorldModelAdapter
from ..config import ALMConfig,FeasibilityConfig,MPCConfig,PlannerConfig,RefinementConfig,SolverConfig
from ..planner import GVPWMPlanner
from ..video import PrecomputedVideoPlanSource
from ..feasibility_model.model import FeasibilityModel
from .dino_oracle_utils import infer_horizon, load_oracle_episode

DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))
from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

DATA_DIR = DINO_WM_ROOT / "data" / "pusht_noise" / "train"
MODEL_NAME = "pusht"
MAX_HORIZON = None




def to_runtime_observation(obs):
    return {
        "visual": torch.as_tensor(obs["visual"], dtype=torch.float32).permute(2, 0, 1) / 255.0,
        "proprio": torch.as_tensor(obs["proprio"], dtype=torch.float32)[..., :4],
    }


def step_env(
    env: gym.Env,
    action: torch.Tensor,
    action_repeat: int,
    primitive_action_dim: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    action_np = action.detach().cpu().numpy().reshape(action_repeat, primitive_action_dim)

    mean_np = action_mean.detach().cpu().numpy()
    std_np = action_std.detach().cpu().numpy()

    primitive_actions = (action_np * std_np) + mean_np

    obs = None
    for primitive_action in primitive_actions:
        step_out = env.step(primitive_action)

        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
        else:
            obs, _, done, _ = step_out
            terminated, truncated = done, False

        if terminated or truncated:
            break

    return to_runtime_observation(obs)

def make_dino_planner_config(
    mode: str,
    horizon: int,
    lambda_video: float = 0.5,
    lambda_feasibility: float = 0.0,
    total_optimization_steps: int = 60,
) -> PlannerConfig:
    if mode == "alm":
        solver_mode = "alm"
        feasibility_enabled = False
        # look at step count 
        outer_steps = 2
        inner_steps = total_optimization_steps // outer_steps
        

    elif mode == "feasibility":
        solver_mode = "feasibility"
        feasibility_enabled = lambda_feasibility > 0.0
        inner_steps = total_optimization_steps 
        outer_steps = 1

    elif mode == "direct":
        solver_mode = "feasibility"
        feasibility_enabled = False
        inner_steps = total_optimization_steps 
        outer_steps = 1

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return PlannerConfig(
        solver=SolverConfig(
            mode=solver_mode,
            inner_steps=inner_steps,
            learning_rate=0.05,
            lambda_video=lambda_video,
            lambda_goal=10.0,
            lambda_action=0.05,
            use_video_init=True,
            use_video_loss=lambda_video > 0.0,
            fix_states_to_video=False,
        ),
        alm=ALMConfig(
            outer_steps=outer_steps,
            rho_init=1.0,
            rho_growth=1.5,
            rho_max=50.0,
        ),
        mpc=MPCConfig(
            horizon=horizon,
            execution_stride=1,
            warm_start=True,
        ),
        refinement=RefinementConfig(
            enabled=True,
            num_samples=32,
            noise_std=0.3,
        ),
        feasibility=FeasibilityConfig(
            enabled=feasibility_enabled,
            diagnostic_only=False,
            use_in_refinement=feasibility_enabled,
            lambda_feasibility=lambda_feasibility,
            noise_level=0.05,
        ),
    )

def load_feasibility_checkpoint(
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> FeasibilityModel | None:
    if checkpoint_path is None:
        return None

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Feasibility checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = FeasibilityModel(
        latent_dim=checkpoint["latent_dim"],
        action_dim=checkpoint["action_dim"],
        history_length=checkpoint["history_length"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        use_layer_norm=checkpoint["use_layer_norm"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model

def evaluate_episode(
    episode_idx: int,
    mode: str,
    lambda_video: float,
    lambda_feasibility: float,
    feasibility_model: FeasibilityModel | None = None,
    seed: int = 0,
    max_horizon: int = 10,
    total_optimization_steps: int = 60,
) -> dict:
    
    EVAL_EPISODE_IDX = episode_idx

    torch.manual_seed(seed)

    model_path = DINO_WM_ROOT / "checkpoints" / "outputs" / MODEL_NAME
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_ckpt,
        model_cfg,
        model_cfg.num_action_repeat,
        device=device,
    )

    episode = load_oracle_episode(DATA_DIR, episode_idx)
    frame_skip = 5
    horizon = infer_horizon(
        episode["length"],
        frame_skip=frame_skip,
        max_horizon=max_horizon,
    )


    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)


    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    velocities = torch.load("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/data/pusht_noise/train/velocities.pth")


    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = velocities[episode_idx, 0].detach().cpu().numpy()

    full_initial_state = np.concatenate([initial_state, initial_velocity], axis=0)
    env.unwrapped._set_state(full_initial_state)

    primitive_action_dim = int(env.action_space.shape[0])   # 2
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)  # 10
    action_repeat = wm_action_dim // primitive_action_dim   # 5

    primitive_low_raw = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    primitive_high_raw = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)

    primitive_low = torch.full((primitive_action_dim,), -3.0, device=device, dtype=torch.float32)
    primitive_high = torch.full((primitive_action_dim,), 3.0, device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)


    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    # extra check
    if mode == "feasibility" and lambda_feasibility > 0.0 and feasibility_model is None:
        raise ValueError(
            "mode='feasibility' with lambda_feasibility > 0 requires a DINO feasibility checkpoint."
        )
    planner = GVPWMPlanner(
        world_model=world_model,
        config=make_dino_planner_config(
            mode=mode,
            horizon=horizon,
            lambda_video=lambda_video,
            lambda_feasibility=lambda_feasibility,
            total_optimization_steps=total_optimization_steps,
        ),
        feasibility_model=feasibility_model,
    )


    print("starting planner")
    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=lambda action: step_env(env, action, action_repeat=action_repeat, primitive_action_dim=primitive_action_dim,
                                        action_mean=action_mean, action_std=action_std,),
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
    )
    print("planner finished")


    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
    print("first 5 expert actions:", episode["actions"][:5])


    goal_state = np.concatenate(
        [
            episode["states"][-1].detach().cpu().numpy(),
            episode["velocities"][-1].detach().cpu().numpy(),
        ],
        axis=0,
    )

    cur_state = np.array(
        [
            env.unwrapped.agent.position[0],
            env.unwrapped.agent.position[1],
            env.unwrapped.block.position[0],
            env.unwrapped.block.position[1],
            env.unwrapped.block.angle,
            env.unwrapped.agent.velocity[0],
            env.unwrapped.agent.velocity[1],
        ],
        dtype=np.float32,
    )

    metrics = env.unwrapped.eval_state(goal_state, cur_state)

    print("eval metrics:", metrics)



    print("agent position:", env.unwrapped.agent.position)
    print("agent velocity:", env.unwrapped.agent.velocity)
    print("block position:", env.unwrapped.block.position)
    print("block angle:", env.unwrapped.block.angle)


    print("DINO-WM Push-T oracle demo")
    print(f"Episode index: {episode_idx}")
    print(f"Oracle length: {episode['length']}")
    print(f"Planning horizon: {horizon}")
    print(f"Executed {result.executed_actions.shape[0]} actions")
    print(f"Final dynamics residual: {result.steps[-1].dynamics_residual_norm:.6f}")

    print(f"Start proprio shape: {episode['start_obs']['proprio'].shape}")
    print(f"Goal proprio shape: {episode['goal_obs']['proprio'].shape}")
    print(f"Executed latent shape: {tuple(result.executed_latents.shape)}")
    print(f"Executed action shape: {tuple(result.executed_actions.shape)}")

    return {
        "episode_idx": episode_idx,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
        "executed_actions": int(result.executed_actions.shape[0]),
        "planning_horizon": int(horizon),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="DINO-WM oracle demo with ALM/direct/feasibility modes.")

    parser.add_argument(
        "--mode",
        type=str,
        default="direct",
        choices=["alm", "direct", "feasibility", "both"],
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-video", type=float, default=0.5)
    parser.add_argument("--lambda-feasibility", type=float, default=0.0)
    parser.add_argument("--feasibility-checkpoint", type=str, default=None)

    # quick tests
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--total-optimization-steps", type=int, default=60)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feasibility_model = load_feasibility_checkpoint(
        args.feasibility_checkpoint,
        device=device,
    )

    if args.mode == "both":
        modes = ["alm", "direct"]

        if args.lambda_feasibility > 0.0:
            modes.append("feasibility")
    else:
        modes = [args.mode]

    results = []

    episode_indices = range(
        args.start_episode,
        args.start_episode + args.num_episodes,
    )

    for episode_idx in episode_indices:
        for mode in modes:
            print(f"\n=== Episode {episode_idx} | Mode: {mode} ===")

            result = evaluate_episode(
                episode_idx=episode_idx,
                mode=mode,
                lambda_video=args.lambda_video,
                lambda_feasibility=args.lambda_feasibility if mode == "feasibility" else 0.0,
                feasibility_model=feasibility_model if mode == "feasibility" else None,
                seed=args.seed,
                max_horizon=args.max_horizon,
                total_optimization_steps=args.total_optimization_steps,
            )

            result["mode"] = mode
            results.append(result)

    print("\n=== Summary ===")
    for result in results:
        print(
            f"episode={result['episode_idx']:04d} | "
            f"mode={result['mode']:12s} | "
            f"success={result['success']} | "
            f"state_dist={result['state_dist']:.6f} | "
            f"dynamics_residual={result['dynamics_residual']:.6f} | "
            f"horizon={result['planning_horizon']} | "
            f"actions={result['executed_actions']}"
        )


if __name__ == "__main__":
    main()