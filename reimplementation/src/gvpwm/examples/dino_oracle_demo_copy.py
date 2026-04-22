from __future__ import annotations

import argparse
import pickle
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

import torch.nn.functional as F
from ..losses import scale_invariant_alignment



DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
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
    proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32)[..., :4]
    proprio = (proprio - PROPRIO_MEAN[: proprio.shape[-1]]) / PROPRIO_STD[: proprio.shape[-1]]

    return {
        "visual": torch.as_tensor(obs["visual"], dtype=torch.float32).permute(2, 0, 1) / 255.0,
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
    diagnostic_inner_interval: int | None = None,
    diagnostic_outer: bool = False,
) -> GVPWMPlanner:
    lambda_action = 0.05 if horizon == 25 else 0.1
    return GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=25,
                outer_steps=25,
                learning_rate=0.05,
                rho_init=1.0,
                rho_growth=1.9,
                rho_max=1_000.0,
                lambda_video=1.0,
                #lambda_video=10,
                lambda_goal=10.0,
                #lambda_action=0,
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


def candidate_episodes(split_dir: Path, horizon: int, frame_skip: int) -> list[int]:
    with open(split_dir / "seq_lengths.pkl", "rb") as handle:
        seq_lengths = pickle.load(handle)
    min_required_length = horizon * frame_skip + 1
    return [idx for idx, length in enumerate(seq_lengths) if int(length) >= min_required_length]


def _oracle_initialize_latents_from_video(
    self,
    current_latent: torch.Tensor,
    video_latents: torch.Tensor,
) -> torch.Tensor:
    latents = video_latents.clone()
    latents[0] = current_latent
    return latents

def _proprio_slice(self, latent: torch.Tensor) -> torch.Tensor:
    if self.world_model.concat_dim == 0:
        return latent[-1]
    visual_dim = int(self.world_model.encoder.emb_dim)
    return latent[..., visual_dim:]



_debug_align_counter = {"n": 0}
    
def _oracle_video_alignment_loss(self, latent, reference):
    visual_loss = scale_invariant_alignment(
        self._visual_slice(latent),
        self._visual_slice(reference),
    )

    proprio_loss = F.mse_loss(
        _proprio_slice(self, latent),
        _proprio_slice(self, reference),
        reduction="mean",
    )

    #if _debug_align_counter["n"] % 500 == 0:
    #    print(
    #        f"[video align] visual={visual_loss.item():.4f} "
    #        f"proprio={proprio_loss.item():.4f} "
    #        f"weighted_proprio={(10.0 * proprio_loss).item():.4f}"
    #    )

    #_debug_align_counter["n"] += 1

    #return visual_loss + 10.0 * proprio_loss
    return visual_loss

def _oracle_goal_loss(self, latent, goal_latent):
    return super(DinoWorldModelAdapter, self).goal_loss(latent, goal_latent)


def evaluate_episode(
    episode_idx: int,
    split: str,
    horizon: int,
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

    split_dir = DATA_ROOT / split
    episode = load_oracle_episode(split_dir, episode_idx)
    episode = slice_oracle_episode(episode, horizon=horizon, frame_skip=FRAME_SKIP)

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

    action_mean = ACTION_MEAN.to(device=device, dtype=torch.float32)
    action_std = ACTION_STD.to(device=device, dtype=torch.float32)
    rel_actions = torch.load(split_dir / "rel_actions.pth").float()
    rel_actions = rel_actions / 100.0
    rel_actions = (rel_actions - ACTION_MEAN) / ACTION_STD
    primitive_low = rel_actions.amin(dim=(0, 1)).to(device=device, dtype=torch.float32)
    primitive_high = rel_actions.amax(dim=(0, 1)).to(device=device, dtype=torch.float32)
    action_low = primitive_low.repeat(action_repeat)
    action_high = primitive_high.repeat(action_repeat)

    DinoWorldModelAdapter.initialize_latents_from_video = _oracle_initialize_latents_from_video
    DinoWorldModelAdapter.video_alignment_loss = _oracle_video_alignment_loss
    #DinoWorldModelAdapter.goal_loss = _oracle_goal_loss

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
        f"starting ALM planner "
        f"(split={split}, horizon={horizon}, I=25, O=25, gamma=1.9, refinement=500x0.3)"
    )
    
    
    # for debug purpose
    goal_state_for_trace = np.concatenate(
        [
            episode["states"][-1].detach().cpu().numpy(),
            episode["velocities"][-1].detach().cpu().numpy(),
        ],
        axis=0,
    )
    trace_counter = {"step": 0}

    def traced_step_fn(action):
        
        action_np = action.detach().cpu().numpy().reshape(action_repeat, primitive_action_dim)
        primitive_actions = (
            action_np * action_std.detach().cpu().numpy()
        ) + action_mean.detach().cpu().numpy()

        planner_macro_disp_px = primitive_actions.sum(axis=0) * 100.0

        expert_start = trace_counter["step"] * FRAME_SKIP
        expert_end = expert_start + FRAME_SKIP
        expert_macro_disp_px = (
            episode["rel_actions"][expert_start:expert_end]
            .detach()
            .cpu()
            .numpy()
            .sum(axis=0)
        )

        macro_disp_px = primitive_actions.sum(axis=0) * 100.0

        agent_pos_before = np.array(
            [
                env.unwrapped.agent.position[0],
                env.unwrapped.agent.position[1],
            ],
            dtype=np.float32,
        )
        block_pos_before = np.array(
            [
                env.unwrapped.block.position[0],
                env.unwrapped.block.position[1],
            ],
            dtype=np.float32,
        )

        to_block = block_pos_before - agent_pos_before
        agent_block_dist = np.linalg.norm(to_block)

        print(
            f"[action trace {trace_counter['step']}] "
            f"macro_disp_px=({macro_disp_px[0]:.1f},{macro_disp_px[1]:.1f}) "
            f"planner_macro_px=({planner_macro_disp_px[0]:.1f},{planner_macro_disp_px[1]:.1f}) "
            f"expert_macro_px=({expert_macro_disp_px[0]:.1f},{expert_macro_disp_px[1]:.1f}) "
            f"to_block=({to_block[0]:.1f},{to_block[1]:.1f}) "
            f"agent_block_dist={agent_block_dist:.1f}"
        )
        agent_vel = np.array(
            [env.unwrapped.agent.velocity[0], env.unwrapped.agent.velocity[1]],
            dtype=np.float32,
        )
        print(f"vel=({agent_vel[0]:.1f},{agent_vel[1]:.1f})")


        t = trace_counter["step"]
        raw0 = t * FRAME_SKIP
        raw1 = raw0 + FRAME_SKIP

        oracle_now = episode["states"][raw0]
        oracle_next = episode["states"][raw1]

        oracle_agent_disp = oracle_next[:2] - oracle_now[:2]
        oracle_block_disp = oracle_next[2:4] - oracle_now[2:4]
        oracle_angle_disp = oracle_next[4] - oracle_now[4]

        expert_macro_raw = episode["rel_actions"][raw0:raw1].float()
        expert_macro_disp = expert_macro_raw.sum(dim=0)

        planner_macro = action.detach().cpu().reshape(action_repeat, primitive_action_dim)
        planner_macro_raw = ((planner_macro * action_std.cpu()) + action_mean.cpu()) * 100.0
        planner_macro_disp = planner_macro_raw.sum(dim=0)

        print(
            f"[time check {t}] raw_range=[{raw0}:{raw1}] "
            f"oracle_agent_now=({oracle_now[0]:.1f},{oracle_now[1]:.1f}) "
            f"oracle_agent_next=({oracle_next[0]:.1f},{oracle_next[1]:.1f}) "
            f"oracle_agent_disp=({oracle_agent_disp[0]:.1f},{oracle_agent_disp[1]:.1f}) "
            f"expert_macro_disp=({expert_macro_disp[0]:.1f},{expert_macro_disp[1]:.1f}) "
            f"planner_macro_disp=({planner_macro_disp[0]:.1f},{planner_macro_disp[1]:.1f}) "
            f"oracle_block_disp=({oracle_block_disp[0]:.1f},{oracle_block_disp[1]:.1f}) "
            f"oracle_angle_disp={oracle_angle_disp:.3f}"
        )



        obs = step_env(
            env,
            action,
            action_repeat=action_repeat,
            primitive_action_dim=primitive_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        )
        
        cur_state_mid = np.array(
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

        agent_diff = np.linalg.norm(goal_state_for_trace[:2] - cur_state_mid[:2])
        block_diff = np.linalg.norm(goal_state_for_trace[2:4] - cur_state_mid[2:4])
        pos_diff = np.linalg.norm(goal_state_for_trace[:4] - cur_state_mid[:4])
        angle_diff = np.abs(goal_state_for_trace[4] - cur_state_mid[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        
        print(f"goal_block=({goal_state_for_trace[2]:.1f},{goal_state_for_trace[3]:.1f},{goal_state_for_trace[4]:.3f}) ")
        print(
            f"[env step {trace_counter['step']}] "
            f"agent=({cur_state_mid[0]:.1f},{cur_state_mid[1]:.1f}) "
            f"block=({cur_state_mid[2]:.1f},{cur_state_mid[3]:.1f}) "
            f"angle={cur_state_mid[4]:.3f} "
            f"agent_diff={agent_diff:.1f} "
            f"block_diff={block_diff:.1f} "
            f"pos_diff={pos_diff:.1f} "
            f"angle_diff={angle_diff:.3f}"
        )
        
        oracle_idx = min(trace_counter["step"] * FRAME_SKIP, episode["states"].shape[0] - 1)
        oracle_state_mid = episode["states"][oracle_idx].detach().cpu().numpy()

        print(
            f"oracle_t_agent=({oracle_state_mid[0]:.1f},{oracle_state_mid[1]:.1f}) "
            f"oracle_t_block=({oracle_state_mid[2]:.1f},{oracle_state_mid[3]:.1f}) "
            f"oracle_t_angle={oracle_state_mid[4]:.3f} "
        )


        trace_counter["step"] += 1
        return obs
    
    print("video_plan length:", len(episode["video_plan"]))
    print("expected macro video length:", horizon + 1)
    print("states length:", episode["states"].shape[0])
    print("rel_actions length:", episode["rel_actions"].shape[0])
    print("frame_skip:", FRAME_SKIP)
    
    
    
    # Debug: check whether visual video alignment is sensitive to oracle block motion.
    debug_macro_indices = [0, 1, 2, 5, 10, 15, 20, 25]
    debug_macro_indices = [i for i in debug_macro_indices if i < len(episode["video_plan"])]

    with torch.no_grad():
        debug_obs = [episode["video_plan"][i] for i in debug_macro_indices]
        encoded_dbg = world_model.encode_sequence(debug_obs).to(device)

        base = encoded_dbg[0]
        for j, macro_i in enumerate(debug_macro_indices):
            raw_i = macro_i * FRAME_SKIP
            st = episode["states"][raw_i]

            visual_loss_from_start = world_model.video_alignment_loss(
                base,
                encoded_dbg[j],
            )

            agent_dist_from_start = torch.linalg.norm(
                episode["states"][raw_i, :2] - episode["states"][0, :2]
            )
            block_dist_from_start = torch.linalg.norm(
                episode["states"][raw_i, 2:4] - episode["states"][0, 2:4]
            )
            angle_diff_from_start = torch.abs(
                episode["states"][raw_i, 4] - episode["states"][0, 4]
            )
            angle_diff_from_start = torch.minimum(
                angle_diff_from_start,
                2 * torch.pi - angle_diff_from_start,
            )

            print(
                f"[ref sensitivity] macro={macro_i} raw={raw_i} "
                f"visual_loss_from_start={float(visual_loss_from_start.detach().cpu()):.6f} "
                f"agent_dist={float(agent_dist_from_start.cpu()):.1f} "
                f"block_dist={float(block_dist_from_start.cpu()):.1f} "
                f"angle_diff={float(angle_diff_from_start.cpu()):.3f} "
                f"agent=({st[0]:.1f},{st[1]:.1f}) "
                f"block=({st[2]:.1f},{st[3]:.1f}) "
                f"angle={st[4]:.3f}"
            )
    ################################################################

    
    
    
    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=traced_step_fn,
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
    )
    print("ALM planner finished")

    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0
    expert_rel_norm = (episode["rel_actions"][:FRAME_SKIP].float() / 100.0 - action_mean.cpu()) / action_std.cpu()

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
    print("first 5 expert actions:", episode["actions"][:FRAME_SKIP])
    print("first 5 expert relative actions (normalized):", expert_rel_norm)

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
    
    agent_diff = np.linalg.norm(goal_state[:2] - cur_state[:2])
    block_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
    pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
    angle_diff = np.abs(goal_state[4] - cur_state[4])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    vel_diff = np.linalg.norm(goal_state[5:] - cur_state[5:])
    print("agent_diff:", agent_diff)
    print("block_diff:", block_diff)
    print("pos_diff:", pos_diff)
    print("angle_diff:", angle_diff)
    print("vel_diff:", vel_diff)
    print("success thresholds: pos_diff < 20, angle_diff < pi/9 =", np.pi / 9)
    
    print(
        f"[ep {episode_idx}] split={split} horizon={horizon} "
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
    parser = argparse.ArgumentParser(description="GVP-WM Oracle evaluation with ALM latent collocation")
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--horizon", type=int, choices=(25, 50, 80), default=DEFAULT_HORIZON, help="Planning horizon")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside the filtered eligible episode list")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of filtered episodes to evaluate")
    parser.add_argument("--debug-inner-every", type=int, default=None, help="Print ALM diagnostics every N inner iterations")
    parser.add_argument("--debug-outer", action="store_true", help="Print diagnostics after every ALM outer iteration")
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    eligible = candidate_episodes(split_dir, horizon=args.horizon, frame_skip=FRAME_SKIP)
    eval_episodes = eligible[args.start_index : args.start_index + args.num_episodes]

    print(f"Evaluating split={args.split} horizon={args.horizon}")
    print(f"Eligible episodes: {len(eligible)}")
    print(f"Selected episode ids: {eval_episodes}")

    results = []
    for idx in eval_episodes:
        print(f"\n=== Episode {idx} ===")
        try:
            results.append(
                evaluate_episode(
                    episode_idx=idx,
                    split=args.split,
                    horizon=args.horizon,
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
