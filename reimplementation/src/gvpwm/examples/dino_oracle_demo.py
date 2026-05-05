from __future__ import annotations

import argparse
import os
import pickle
import random
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
from .dino_oracle_utils import load_oracle_episode, slice_oracle_episode


DINO_WM_ROOT = Path(
    os.environ.get("DINO_WM_ROOT", "/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
)
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD


DATA_ROOT = DINO_WM_ROOT / "data" / "pusht_noise"
MODEL_NAME = "pusht"
FRAME_SKIP = 5

DEFAULT_HORIZON = 25
DEFAULT_SPLIT = "val"
DEFAULT_NUM_EPISODES = 50


def to_runtime_observation(obs):
    visual = torch.as_tensor(obs["visual"], dtype=torch.float32).permute(2, 0, 1) / 255.0
    visual = (visual - 0.5) / 0.5
    raw_proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32)[..., :4]
    proprio = (raw_proprio - PROPRIO_MEAN[:4]) / PROPRIO_STD[:4]
    return {
        "visual": visual,
        "proprio": proprio,
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
    primitive_actions = (action_np * action_std.detach().cpu().numpy()) + action_mean.detach().cpu().numpy()

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


def load_model_once(device):
    model_path = DINO_WM_ROOT / "checkpoints" / "outputs" / MODEL_NAME
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    return model, model_cfg


def build_planner(
    world_model: DinoWorldModelAdapter,
    horizon: int,
    paper_horizon: int | None = None,
    inner_steps: int = 25,
    outer_steps: int = 25,
    lambda_action_override: float | None = None,
    lambda_action_prior: float = 0.0,
    lambda_goal: float = 10.0,
    lambda_video: float = 1.0,
    residual_reduction: str = "sum",
    fix_states_to_video: bool = False,
    use_action_reparameterization: bool = True,
    refinement_samples: int = 500,
    refinement_variance: float = 0.3,
    disable_refinement: bool = False,
    diagnostic_inner_interval: int | None = None,
    diagnostic_outer: bool = False,
) -> GVPWMPlanner:
    hyperparam_horizon = paper_horizon if paper_horizon is not None else horizon
    lambda_action = 0.05 if hyperparam_horizon == 25 else 0.1
    if lambda_action_override is not None:
        lambda_action = lambda_action_override
    return GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=inner_steps,
                outer_steps=outer_steps,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=1.9,
                rho_max=1_000.0,
                lambda_video=lambda_video,
                lambda_goal=lambda_goal,
                lambda_action=lambda_action,
                lambda_action_prior=lambda_action_prior,
                use_video_init=True,
                use_video_loss=True,
                fix_states_to_video=fix_states_to_video,
                use_action_reparameterization=use_action_reparameterization,
                diagnostic_inner_interval=diagnostic_inner_interval,
                diagnostic_outer=diagnostic_outer,
                residual_reduction=residual_reduction,
            ),
            mpc=MPCConfig(
                horizon=horizon,
                execution_stride=1,
                warm_start=True,
            ),
            refinement=RefinementConfig(
                enabled=not disable_refinement,
                num_samples=refinement_samples,
                noise_variance=refinement_variance,
            ),
        ),
    )


def candidate_episodes(split_dir: Path, horizon: int, frame_skip: int) -> list[int]:
    with open(split_dir / "seq_lengths.pkl", "rb") as handle:
        seq_lengths = pickle.load(handle)
    min_required_length = horizon * frame_skip + 1
    return [idx for idx, length in enumerate(seq_lengths) if int(length) >= min_required_length]


def parse_episode_specs(value: str) -> list[tuple[int, int]]:
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            episode_text, offset_text = item.split(":", 1)
            specs.append((int(episode_text), int(offset_text)))
        else:
            specs.append((int(item), 0))
    return specs


def sample_segment_specs(
    split_dir: Path,
    horizon: int,
    frame_skip: int,
    num_segments: int,
    seed: int,
    start_index: int = 0,
) -> list[tuple[int, int]]:
    with open(split_dir / "seq_lengths.pkl", "rb") as handle:
        seq_lengths = [int(length) for length in pickle.load(handle)]
    required_frames = horizon * frame_skip + 1
    rng = random.Random(seed)
    sampled = []
    total_needed = start_index + num_segments
    while len(sampled) < total_needed:
        episode_idx = rng.randint(0, len(seq_lengths) - 1)
        max_offset = seq_lengths[episode_idx] - required_frames
        if max_offset < 0:
            continue
        sampled.append((episode_idx, rng.randint(0, max_offset)))
    return sampled[start_index:total_needed]


def _oracle_initialize_latents_from_video(
    self,
    current_latent: torch.Tensor,
    video_latents: torch.Tensor,
) -> torch.Tensor:
    latents = video_latents.clone()
    latents[0] = current_latent
    return latents

def _oracle_video_alignment_loss(self, latent, reference):
    return super(DinoWorldModelAdapter, self).video_alignment_loss(latent, reference)

def _oracle_goal_loss(self, latent, goal_latent):
    return super(DinoWorldModelAdapter, self).goal_loss(latent, goal_latent)


def evaluate_episode(
    episode_idx: int,
    start_offset: int,
    split: str,
    horizon: int,
    frame_skip: int,
    paper_horizon: int | None = None,
    inner_steps: int = 25,
    outer_steps: int = 25,
    lambda_action_override: float | None = None,
    lambda_action_prior: float = 0.0,
    lambda_goal: float = 10.0,
    lambda_video: float = 1.0,
    residual_reduction: str = "mean",
    fix_states_to_video: bool = False,
    use_action_reparameterization: bool = True,
    refinement_samples: int = 500,
    refinement_variance: float = 0.3,
    disable_refinement: bool = False,
    expert_action_warmstart: bool = False,
    visual_only_guidance: bool = True,
    diagnostic_inner_interval: int | None = None,
    diagnostic_outer: bool = False,
    model=None,
    model_cfg=None,
    device=None,
) -> dict:
    if paper_horizon is None:
        paper_horizon = horizon * frame_skip
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or model_cfg is None:
        model, model_cfg = load_model_once(device)

    torch.manual_seed(episode_idx * 100_000 + start_offset)

    split_dir = DATA_ROOT / split
    episode = load_oracle_episode(split_dir, episode_idx)
    episode = slice_oracle_episode(
        episode,
        horizon=horizon,
        frame_skip=frame_skip,
        start_offset=start_offset,
    )

    env = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _ = reset_out[0]

    initial_state = episode["states"][0].detach().cpu().numpy()
    initial_velocity = episode["velocities"][0].detach().cpu().numpy()
    full_initial_state = np.concatenate([initial_state, initial_velocity], axis=0)
    env.unwrapped._set_state(full_initial_state)

    primitive_action_dim = int(env.action_space.shape[0])
    wm_action_dim = int(model.action_encoder.patch_embed.in_channels)
    action_repeat = wm_action_dim // primitive_action_dim
    if wm_action_dim % primitive_action_dim != 0:
        raise ValueError(
            f"World-model action dim {wm_action_dim} is not divisible by "
            f"PushT primitive action dim {primitive_action_dim}."
        )
    if action_repeat != frame_skip:
        raise ValueError(
            f"Invalid PushT time scale: action_repeat={action_repeat}, frame_skip={frame_skip}. "
            "Use --frame-skip matching the checkpoint action concatenation."
        )

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)
    rel_actions = torch.load(split_dir / "rel_actions.pth").float()
    rel_actions = rel_actions / 100.0
    rel_actions = (rel_actions - ACTION_MEAN) / ACTION_STD
    primitive_low = rel_actions.amin(dim=(0, 1)).to(device=device, dtype=torch.float32)
    primitive_high = rel_actions.amax(dim=(0, 1)).to(device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)

    if not visual_only_guidance:
        # Diagnostic upper bound only: this injects oracle proprio latents from
        # the demonstration. Paper-faithful evaluation keeps guidance visual-only.
        DinoWorldModelAdapter.initialize_latents_from_video = _oracle_initialize_latents_from_video
        DinoWorldModelAdapter.video_alignment_loss = _oracle_video_alignment_loss
        DinoWorldModelAdapter.goal_loss = _oracle_goal_loss

    world_model = DinoWorldModelAdapter(
        world_model=model,
        action_dim=wm_action_dim,
        action_low=action_low,
        action_high=action_high,
    )
    planner = build_planner(
        world_model=world_model,
        horizon=horizon,
        paper_horizon=paper_horizon,
        inner_steps=inner_steps,
        outer_steps=outer_steps,
        lambda_action_override=lambda_action_override,
        lambda_action_prior=lambda_action_prior,
        lambda_goal=lambda_goal,
        lambda_video=lambda_video,
        residual_reduction=residual_reduction,
        fix_states_to_video=fix_states_to_video,
        use_action_reparameterization=use_action_reparameterization,
        refinement_samples=refinement_samples,
        refinement_variance=refinement_variance,
        disable_refinement=disable_refinement,
        diagnostic_inner_interval=diagnostic_inner_interval,
        diagnostic_outer=diagnostic_outer,
    )

    print(
        f"starting ALM planner "
        f"(split={split}, macro_horizon={horizon}, raw_horizon={paper_horizon}, "
        f"frame_skip={frame_skip}, I={inner_steps}, O={outer_steps}, gamma=1.9, "
        f"lambda_action={planner.config.alm.lambda_action}, "
        f"lambda_action_prior={planner.config.alm.lambda_action_prior}, "
        f"lambda_goal={planner.config.alm.lambda_goal}, "
        f"lambda_video={planner.config.alm.lambda_video}, "
        f"residual_reduction={planner.config.alm.residual_reduction}, "
        f"fix_states_to_video={planner.config.alm.fix_states_to_video}, "
        f"action_reparam={planner.config.alm.use_action_reparameterization}, "
        f"expert_action_warmstart={expert_action_warmstart}, "
        f"visual_only_guidance={visual_only_guidance}, "
        f"refinement={'off' if disable_refinement else f'{refinement_samples}x{refinement_variance}'})"
    )
    initial_warm_start_actions = None
    if expert_action_warmstart:
        expert_actions = episode["rel_actions"][: horizon * frame_skip].float().to(device) / 100.0
        expert_actions = (expert_actions - action_mean) / action_std
        initial_warm_start_actions = expert_actions.reshape(horizon, action_repeat * primitive_action_dim)

    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=lambda action: step_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        ),
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
        initial_warm_start_actions=initial_warm_start_actions,
    )
    print("ALM planner finished")

    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0
    expert_rel_norm = (
        episode["rel_actions"][:frame_skip].float() / 100.0 - action_mean.cpu()
    ) / action_std.cpu()

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
    print(f"first {frame_skip} expert actions:", episode["actions"][:frame_skip])
    print(f"first {frame_skip} expert relative actions (normalized):", expert_rel_norm)

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
    agent_diff = np.linalg.norm(goal_state[:2] - cur_state[:2])
    block_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
    angle_diff = np.abs(
        (float(goal_state[4]) - float(cur_state[4]) + np.pi) % (2 * np.pi) - np.pi
    )
    vel_diff = np.linalg.norm(goal_state[5:] - cur_state[5:])
    block_success = block_diff < 20 and angle_diff < np.pi / 9

    print("eval metrics (env legacy):", metrics)
    print("block-pose success:", block_success)
    print("agent position:", env.unwrapped.agent.position)
    print("agent velocity:", env.unwrapped.agent.velocity)
    print("block position:", env.unwrapped.block.position)
    print("block angle:", env.unwrapped.block.angle)
    print("agent_diff:", agent_diff)
    print("block_diff:", block_diff)
    print("angle_diff:", angle_diff)
    print("vel_diff:", vel_diff)
    print("success thresholds: block_diff < 20, angle_diff < pi/9 =", np.pi / 9)
    
    print(
        f"[ep {episode_idx}] split={split} horizon={horizon} "
        f"raw_horizon={paper_horizon} frame_skip={frame_skip} offset={start_offset} "
        f"success={block_success} block_diff={block_diff:.2f} "
        f"angle_diff={angle_diff:.3f} "
        f"dyn_residual={result.steps[-1].dynamics_residual_norm:.4f}"
    )

    return {
        "episode_idx": episode_idx,
        "start_offset": int(start_offset),
        "split": split,
        "success": bool(block_success),
        "state_dist": float(block_diff),
        "legacy_env_success": bool(metrics["success"]),
        "legacy_env_state_dist": float(metrics["state_dist"]),
        "angle_diff": float(angle_diff),
        "dynamics_residual": float(result.steps[-1].dynamics_residual_norm),
        "executed_actions": int(result.executed_actions.shape[0]),
        "planning_horizon": int(horizon),
        "raw_horizon": int(paper_horizon if paper_horizon is not None else horizon * frame_skip),
        "frame_skip": int(frame_skip),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GVP-WM Oracle evaluation with ALM latent collocation")
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="World-model macro planning horizon")
    parser.add_argument("--raw-horizon", type=int, default=None, help="Paper/environment horizon T; overrides --horizon via T / frame_skip")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help="Raw environment steps per world-model step")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside the filtered eligible episode list")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of filtered episodes to evaluate")
    parser.add_argument("--episode-ids", default=None, help="Comma-separated explicit episode ids to evaluate")
    parser.add_argument(
        "--episode-specs",
        default=None,
        help="Comma-separated episode[:offset] specs. Overrides --episode-ids when provided.",
    )
    parser.add_argument(
        "--sample-targets",
        action="store_true",
        help="Sample DINO-WM-style trajectory segments with replacement instead of using offset 0.",
    )
    parser.add_argument("--sample-seed", type=int, default=99, help="Seed for --sample-targets segment sampling")
    parser.add_argument("--inner-steps", type=int, default=25, help="ALM inner optimizer steps")
    parser.add_argument("--outer-steps", type=int, default=25, help="ALM outer penalty updates")
    parser.add_argument("--lambda-action", type=float, default=None, help="Override ALM action regularization weight")
    parser.add_argument(
        "--lambda-action-prior",
        type=float,
        default=0.0,
        help="Penalty weight for staying near warm-start actions.",
    )
    parser.add_argument("--lambda-goal", type=float, default=10.0, help="ALM goal loss weight")
    parser.add_argument("--lambda-video", type=float, default=1.0, help="ALM video alignment loss weight")
    parser.add_argument(
        "--residual-reduction",
        choices=("mean", "sum"),
        default="sum",
        help="Scale the ALM dynamics penalty by latent dimensionality ('mean') or use paper-style sum.",
    )
    parser.add_argument(
        "--fix-states-to-video",
        action="store_true",
        help="Keep collocation latents fixed to oracle video latents while solving actions.",
    )
    parser.add_argument(
        "--disable-action-reparameterization",
        action="store_true",
        help="Optimize normalized actions directly with clamp instead of tanh reparameterization.",
    )
    parser.add_argument("--refinement-samples", type=int, default=500, help="Number of random refinement samples")
    parser.add_argument("--refinement-variance", type=float, default=0.3, help="Refinement noise variance")
    parser.add_argument("--disable-refinement", action="store_true", help="Disable random action refinement")
    parser.add_argument(
        "--expert-action-warmstart",
        action="store_true",
        help="Diagnostic only: initialize the first MPC solve from dataset expert actions.",
    )
    parser.add_argument(
        "--visual-only-guidance",
        action="store_true",
        help="Deprecated compatibility flag; visual-only guidance is now the default.",
    )
    parser.add_argument(
        "--oracle-proprio-guidance",
        action="store_true",
        help="Diagnostic only: include oracle proprio latents in video/goal losses.",
    )
    parser.add_argument("--quick", action="store_true", help="Use small ALM/refinement settings for smoke tests")
    parser.add_argument("--debug-inner-every", type=int, default=None, help="Print ALM diagnostics every N inner iterations")
    parser.add_argument("--debug-outer", action="store_true", help="Print diagnostics after every ALM outer iteration")
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    frame_skip = int(args.frame_skip)
    if args.raw_horizon is not None:
        if args.raw_horizon % frame_skip != 0:
            raise ValueError(
                f"--raw-horizon {args.raw_horizon} must be divisible by --frame-skip {frame_skip}."
            )
        macro_horizon = args.raw_horizon // frame_skip
        paper_horizon = args.raw_horizon
    else:
        macro_horizon = args.horizon
        paper_horizon = macro_horizon * frame_skip

    split_dir = DATA_ROOT / args.split
    eligible = candidate_episodes(split_dir, horizon=macro_horizon, frame_skip=frame_skip)
    if args.episode_specs:
        eval_specs = parse_episode_specs(args.episode_specs)
    elif args.sample_targets:
        eval_specs = sample_segment_specs(
            split_dir,
            horizon=macro_horizon,
            frame_skip=frame_skip,
            num_segments=args.num_episodes,
            seed=args.sample_seed,
            start_index=args.start_index,
        )
    elif args.episode_ids:
        eval_specs = [(int(item), 0) for item in args.episode_ids.split(",") if item.strip()]
    else:
        eval_specs = [
            (idx, 0)
            for idx in eligible[args.start_index : args.start_index + args.num_episodes]
        ]

    inner_steps = 3 if args.quick else args.inner_steps
    outer_steps = 2 if args.quick else args.outer_steps
    refinement_samples = 0 if args.quick else args.refinement_samples
    disable_refinement = args.disable_refinement or args.quick

    print(
        f"Evaluating split={args.split} macro_horizon={macro_horizon} "
        f"raw_horizon={paper_horizon} frame_skip={frame_skip}"
    )
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode specs: {eval_specs}")

    results = []
    for idx, start_offset in eval_specs:
        print(f"\n=== Episode {idx} offset {start_offset} ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    start_offset=start_offset,
                    split=args.split,
                    horizon=macro_horizon,
                    frame_skip=frame_skip,
                    paper_horizon=paper_horizon,
                    inner_steps=inner_steps,
                    outer_steps=outer_steps,
                    lambda_action_override=args.lambda_action,
                    lambda_action_prior=args.lambda_action_prior,
                    lambda_goal=args.lambda_goal,
                    lambda_video=args.lambda_video,
                    residual_reduction=args.residual_reduction,
                    fix_states_to_video=args.fix_states_to_video,
                    use_action_reparameterization=not args.disable_action_reparameterization,
                    refinement_samples=refinement_samples,
                    refinement_variance=args.refinement_variance,
                    disable_refinement=disable_refinement,
                    expert_action_warmstart=args.expert_action_warmstart,
                    visual_only_guidance=args.visual_only_guidance or not args.oracle_proprio_guidance,
                    diagnostic_inner_interval=args.debug_inner_every,
                    diagnostic_outer=args.debug_outer,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                )
            )
        except Exception as exc:
            import traceback

            print(f"[ep {idx}] ERROR: {exc}")
            traceback.print_exc()

    print("\n=== Evaluation Summary ===")
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
