from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gym
import numpy as np
import torch
from omegaconf import OmegaConf

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig
from ..planner import GVPWMPlanner
from ..video import PrecomputedVideoPlanSource
from .dino_wall_oracle_utils import (
    candidate_wall_episodes,
    compute_wall_stats,
    load_wall_oracle_episode,
    make_wall_observation,
    resolve_wall_data_dir,
    slice_wall_oracle_episode,
)


DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

import env as _dino_env  # noqa: F401  # registers "wall" with gym
from plan import load_model


DATA_ROOT = DINO_WM_ROOT / "data" / "wall_single"
MODEL_NAME = "wall_single"

DEFAULT_HORIZON = 25
DEFAULT_SPLIT = "all"
DEFAULT_NUM_EPISODES = 50


def to_runtime_observation(
    obs: dict,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw_proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32)
    mean = proprio_mean.to(device=raw_proprio.device, dtype=raw_proprio.dtype)
    std = proprio_std.to(device=raw_proprio.device, dtype=raw_proprio.dtype)
    proprio = (raw_proprio - mean) / std
    return make_wall_observation(obs["visual"], proprio)


def step_env(
    env: gym.Env,
    action: torch.Tensor,
    action_repeat: int,
    primitive_action_dim: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    actions = action.detach().reshape(action_repeat, primitive_action_dim)
    primitive_actions = actions * action_std.to(actions.device) + action_mean.to(actions.device)

    obs = None
    env_device = torch.device(getattr(env.unwrapped, "device", "cpu"))
    for primitive_action in primitive_actions:
        primitive_action = primitive_action.to(device=env_device, dtype=torch.float32)
        step_out = env.step(primitive_action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
        else:
            obs, _, done, _ = step_out
            terminated, truncated = done, False
        if terminated or truncated:
            break

    if obs is None:
        raise RuntimeError("Wall environment did not return an observation.")

    if hasattr(env.unwrapped, "transform"):
        obs = dict(obs)
        obs["visual"] = env.unwrapped.transform(obs["visual"]).permute(1, 2, 0)

    return to_runtime_observation(obs, proprio_mean=proprio_mean, proprio_std=proprio_std)


def load_model_once(device: torch.device, model_name: str = MODEL_NAME):
    model_path = DINO_WM_ROOT / "checkpoints" / "outputs" / model_name
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    return model, model_cfg


def build_planner(
    world_model: DinoWorldModelAdapter,
    horizon: int,
    diagnostic_inner_interval: int | None = None,
    diagnostic_outer: bool = False,
) -> GVPWMPlanner:
    lambda_action = 0.05 if horizon == 25 else 0.1
    rho_growth = 1.5 if horizon == 25 else 1.9
    
    
    
    """
            alm=ALMConfig(
                inner_steps=25,
                outer_steps=25,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=rho_growth,
                rho_max=1_000.0,
                lambda_video=1.0,
                lambda_goal=10.0,
                lambda_action=lambda_action,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=False,
                use_action_reparameterization=True,
                diagnostic_inner_interval=diagnostic_inner_interval,
                diagnostic_outer=diagnostic_outer,
                residual_reduction="mean",
            ),
    """
    
    
    return GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=25,
                outer_steps=25,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=rho_growth,
                rho_max=1_000.0,
                lambda_video=1.0,
                lambda_goal=10.0,
                lambda_action=lambda_action,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=False,
                use_action_reparameterization=True,
                diagnostic_inner_interval=diagnostic_inner_interval,
                diagnostic_outer=diagnostic_outer,
                residual_reduction="mean",
            ),
            mpc=MPCConfig(
                horizon=horizon,
                execution_stride=1,
                warm_start=True,
            ),
            refinement=RefinementConfig(
                enabled=True,
                num_samples=500,
                noise_variance=0.3,
            ),
        ),
    )


def _oracle_initialize_latents_from_video(
    self,
    current_latent: torch.Tensor,
    video_latents: torch.Tensor,
) -> torch.Tensor:
    latents = video_latents.clone()
    latents[0] = current_latent
    return latents


def _prepare_wall_env(env: gym.Env, episode: dict, episode_idx: int) -> None:
    env.unwrapped.update_env(episode["env_info"])
    init_state = episode["states"][0].detach().cpu().numpy()
    env.unwrapped.seed(episode_idx)
    env.unwrapped.set_init_state(init_state)
    env.reset()


def _current_wall_state(env: gym.Env) -> np.ndarray:
    return env.unwrapped.dot_position.detach().cpu().numpy().astype(np.float32)


def evaluate_episode(
    episode_idx: int,
    split: str,
    horizon: int,
    frame_skip: int,
    data_dir: Path,
    stats: dict[str, torch.Tensor],
    diagnostic_inner_interval: int | None = None,
    diagnostic_outer: bool = False,
    model=None,
    model_cfg=None,
    device=None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or model_cfg is None:
        model, model_cfg = load_model_once(device)

    torch.manual_seed(episode_idx)

    episode = load_wall_oracle_episode(data_dir, episode_idx, stats=stats)
    episode = slice_wall_oracle_episode(episode, horizon=horizon, frame_skip=frame_skip)

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    _prepare_wall_env(env, episode, episode_idx)

    primitive_action_dim = int(getattr(env.unwrapped, "action_dim", 2))
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    if wm_action_dim % primitive_action_dim != 0:
        raise ValueError(
            f"World-model action dim {wm_action_dim} is not divisible by "
            f"Wall primitive action dim {primitive_action_dim}."
        )
    action_repeat = wm_action_dim // primitive_action_dim

    action_mean = episode["action_mean"].to(device=device, dtype=torch.float32)
    action_std = episode["action_std"].to(device=device, dtype=torch.float32)
    action_low = episode["action_low"].to(device=device, dtype=torch.float32).repeat(action_repeat)
    action_high = episode["action_high"].to(device=device, dtype=torch.float32).repeat(action_repeat)

    DinoWorldModelAdapter.initialize_latents_from_video = _oracle_initialize_latents_from_video

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    planner = build_planner(
        world_model=world_model,
        horizon=horizon,
        diagnostic_inner_interval=diagnostic_inner_interval,
        diagnostic_outer=diagnostic_outer,
    )

    print(
        f"starting Wall ALM planner "
        f"(split={split}, horizon={horizon}, frame_skip={frame_skip}, "
        f"I=25, O=25, gamma={'1.5' if horizon == 25 else '1.9'}, refinement=500x0.3)"
    )

    trace_counter = {"step": 0}
    goal_state = episode["states"][-1].detach().cpu().numpy()

    def traced_step_fn(action: torch.Tensor):
        t = trace_counter["step"]
        raw0 = t * frame_skip
        raw1 = raw0 + frame_skip

        action_matrix = action.detach().cpu().reshape(action_repeat, primitive_action_dim)
        planner_macro_action = ((action_matrix * action_std.cpu()) + action_mean.cpu()).sum(dim=0)
        planner_delta_est = planner_macro_action * 2.0

        expert_macro_action = episode["actions"][raw0:raw1].float().sum(dim=0)
        expert_delta_est = expert_macro_action * 2.0

        oracle_now = episode["states"][raw0]
        oracle_next = episode["states"][raw1]
        oracle_disp = oracle_next - oracle_now

        cur_before = _current_wall_state(env)
        to_goal_before = goal_state - cur_before

        print(
            f"[wall action trace {t}] "
            f"planner_action_sum=({planner_macro_action[0]:.3f},{planner_macro_action[1]:.3f}) "
            f"planner_delta_est=({planner_delta_est[0]:.2f},{planner_delta_est[1]:.2f}) "
            f"expert_action_sum=({expert_macro_action[0]:.3f},{expert_macro_action[1]:.3f}) "
            f"expert_delta_est=({expert_delta_est[0]:.2f},{expert_delta_est[1]:.2f}) "
            f"to_goal=({to_goal_before[0]:.2f},{to_goal_before[1]:.2f}) "
            f"goal_dist={np.linalg.norm(to_goal_before):.2f}"
        )
        print(
            f"[wall time check {t}] raw_range=[{raw0}:{raw1}] "
            f"oracle_now=({oracle_now[0]:.2f},{oracle_now[1]:.2f}) "
            f"oracle_next=({oracle_next[0]:.2f},{oracle_next[1]:.2f}) "
            f"oracle_disp=({oracle_disp[0]:.2f},{oracle_disp[1]:.2f})"
        )

        obs = step_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
            proprio_mean=episode["proprio_mean"],
            proprio_std=episode["proprio_std"],
        )

        cur_state = _current_wall_state(env)
        state_diff = np.linalg.norm(goal_state - cur_state)
        oracle_idx = min(t * frame_skip, episode["states"].shape[0] - 1)
        oracle_state = episode["states"][oracle_idx].detach().cpu().numpy()
        print(
            f"[wall env step {t}] "
            f"state=({cur_state[0]:.2f},{cur_state[1]:.2f}) "
            f"goal=({goal_state[0]:.2f},{goal_state[1]:.2f}) "
            f"state_diff={state_diff:.2f} "
            f"oracle_t=({oracle_state[0]:.2f},{oracle_state[1]:.2f})"
        )

        trace_counter["step"] += 1
        return obs

    print("video_plan length:", len(episode["video_plan"]))
    print("expected macro video length:", horizon + 1)
    print("states length:", episode["states"].shape[0])
    print("actions length:", episode["actions"].shape[0])
    print("frame_skip:", frame_skip)
    print(
        f"wall layout: door={float(episode['env_info']['fix_door_location']):.2f} "
        f"wall={float(episode['env_info']['fix_wall_location']):.2f}"
    )

    debug_macro_indices = [0, 1, 2, 5, 10, 15, 20, horizon]
    debug_macro_indices = sorted({i for i in debug_macro_indices if i < len(episode["video_plan"])})
    with torch.no_grad():
        debug_obs = [episode["video_plan"][i] for i in debug_macro_indices]
        encoded_dbg = world_model.encode_sequence(debug_obs).to(device)
        base = encoded_dbg[0]
        for j, macro_i in enumerate(debug_macro_indices):
            raw_i = macro_i * frame_skip
            st = episode["states"][raw_i]
            visual_loss_from_start = world_model.video_alignment_loss(base, encoded_dbg[j])
            state_dist_from_start = torch.linalg.norm(episode["states"][raw_i] - episode["states"][0])
            print(
                f"[wall ref sensitivity] macro={macro_i} raw={raw_i} "
                f"visual_loss_from_start={float(visual_loss_from_start.detach().cpu()):.6f} "
                f"state_dist={float(state_dist_from_start.cpu()):.2f} "
                f"state=({st[0]:.2f},{st[1]:.2f})"
            )

    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=traced_step_fn,
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
    )
    print("Wall ALM planner finished")

    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = (first_macro * action_std.cpu()) + action_mean.cpu()
    expert_actions_norm = episode["actions_normalized"][:frame_skip]

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env action scale):", first_macro_raw)
    print("first expert raw actions:", episode["actions"][:frame_skip])
    print("first expert normalized actions:", expert_actions_norm)

    cur_state = _current_wall_state(env)
    metrics = env.unwrapped.eval_state(goal_state, cur_state)

    print("eval metrics:", metrics)
    print("agent/dot position:", cur_state)
    print("goal position:", goal_state)
    print("state_diff:", np.linalg.norm(goal_state - cur_state))
    print("success threshold: state_dist < 4.5")
    print(
        f"[wall ep {episode_idx}] split={split} horizon={horizon} "
        f"success={metrics['success']} state_dist={metrics['state_dist']:.2f} "
        f"dyn_residual={result.steps[-1].dynamics_residual_norm:.4f}"
    )

    return {
        "episode_idx": episode_idx,
        "split": split,
        "success": bool(metrics["success"]),
        "state_dist": float(metrics["state_dist"]),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
        "executed_actions": int(result.executed_actions.shape[0]),
        "planning_horizon": int(horizon),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GVP-WM Wall oracle evaluation with ALM latent collocation")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Use 'all' for wall_single, or a folder split if present.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Planning horizon in macro steps")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override model/data frame skip")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside the eligible episode list")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of eligible episodes to evaluate")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Checkpoint folder under dino_wm/checkpoints/outputs")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Wall dataset directory")
    parser.add_argument("--debug-inner-every", type=int, default=None, help="Print ALM diagnostics every N inner iterations")
    parser.add_argument("--debug-outer", action="store_true", help="Print diagnostics after every ALM outer iteration")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device, model_name=args.model_name)

    frame_skip = args.frame_skip
    if frame_skip is None:
        frame_skip = int(getattr(model_cfg, "frameskip", 1))

    data_dir = resolve_wall_data_dir(args.data_root, args.split)
    stats = compute_wall_stats(data_dir)
    eligible = candidate_wall_episodes(data_dir, horizon=args.horizon, frame_skip=frame_skip)
    eval_episodes = eligible[args.start_index : args.start_index + args.num_episodes]

    print(f"Evaluating Wall split={args.split} data_dir={data_dir} horizon={args.horizon}")
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode ids: {eval_episodes}")

    results = []
    for idx in eval_episodes:
        print(f"\n=== Wall Episode {idx} ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    split=args.split,
                    horizon=args.horizon,
                    frame_skip=frame_skip,
                    data_dir=data_dir,
                    stats=stats,
                    diagnostic_inner_interval=args.debug_inner_every,
                    diagnostic_outer=args.debug_outer,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                )
            )
        except Exception as exc:
            import traceback

            print(f"[wall ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Wall Evaluation Summary ===")
    for result in results:
        print(result)
    if results:
        success_rate = sum(result["success"] for result in results) / len(results)
        mean_dist = sum(result["state_dist"] for result in results) / len(results)
        mean_dyn = sum(result["dynamics_residual"] for result in results) / len(results)
        print(
            f"Success rate: {success_rate:.3f}  "
            f"mean_state_dist: {mean_dist:.2f}  "
            f"mean_dyn_res: {mean_dyn:.4f}"
        )


if __name__ == "__main__":
    main()
