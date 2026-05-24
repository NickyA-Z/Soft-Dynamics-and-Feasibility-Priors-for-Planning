from __future__ import annotations

import argparse
from html import parser
import pickle
import sys
from pathlib import Path

import gym
import numpy as np
from reimplementation.src.gvpwm.examples.dino_oracle_diagnose import expert_macro_actions
import torch
from omegaconf import OmegaConf

from reimplementation.src.gvpwm.adapters.dino_wm import DinoWorldModelAdapter
from reimplementation.src.gvpwm.config import ALMConfig, MPCConfig, PlannerConfig, RefinementConfig, FeasibilityConfig
from reimplementation.src.gvpwm.planner import GVPWMPlanner
from reimplementation.src.gvpwm.video import PrecomputedVideoPlanSource
from reimplementation.src.gvpwm.examples.dino_oracle_utils import load_oracle_episode, slice_oracle_episode

import torch.nn.functional as F
from reimplementation.src.gvpwm.losses import scale_invariant_alignment



DINO_WM_ROOT = Path("/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm")
if str(DINO_WM_ROOT) not in sys.path:
    sys.path.append(str(DINO_WM_ROOT))

from plan import load_model
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD

DATA_ROOT = DINO_WM_ROOT / "data" / "pusht_noise"
MODEL_NAME = "pusht"
FRAME_SKIP = 5

DEFAULT_RAW_HORIZON = 25
DEFAULT_SPLIT = "val"
DEFAULT_NUM_EPISODES = 50


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


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

# cleaned config
def build_planner(
    world_model: DinoWorldModelAdapter,
    horizon: int,
    feasibility_checkpoint: str,  # Path to trained feasibility checkpoint, passed from CLI/sbatch.
    # new added 19 may
    dynamics_mode: str,
    use_dynamics_constraints: bool,
    lambda_dynamics: float,
    lambda_anti_stillness: float,
    min_action_norm: float,
    feasibility_enabled: bool,
    lambda_feasibility: float,
    lambda_transition: float,
    lambda_action_consistency: float,

    diagnostic_inner_interval: int | None = None,  # Print solver diagnostics every N inner steps. None disables inner logging.
    diagnostic_outer: bool = False,  # Whether to print diagnostics after each outer solver loop.

) -> GVPWMPlanner:
    # Penalizes large actions. Smaller horizon gets slightly lower penalty.
    # If too high, actions stay tiny. If too low, actions may become noisy/large.
    return GVPWMPlanner(
        world_model=world_model,
        config=PlannerConfig(
            alm=ALMConfig(
                inner_steps=25, #(22749227:25))
                outer_steps=1,
                learning_rate=0.01,  # Adam step size for optimizing latent/action variables.
                rho_init=1.0,
                rho_growth=1.9,
                rho_max=1_000.0,
                lambda_video=1.0,  # Weight for matching intermediate latents to the reference/oracle video.
                lambda_goal=10.0,  # Weight for matching final latent to the goal latent.
                lambda_action=0.0,  # Weight for action L2 regularization: actions.pow(2).sum().
                lambda_action_prior=0.0,  # Optional weight for staying close to warm-start/prior actions.
                clip_grad_norm=None,  # If set, clips gradient norm of optimized action/latent parameters.
                use_video_init=True,  # If True, initialize latent trajectory from video latents; if False, interpolate current->goal.
                use_video_loss=True,  # If True, include video alignment loss for intermediate latents.
                fix_states_to_video=False,  # If True, fix latents and only optimize actions.
                use_action_reparameterization=True,  # (22749227:True)If True, optimize unconstrained params through tanh into action bounds.
                adam_eps=1e-8,  # Numerical epsilon used by Adam optimizer.
                diagnostic_inner_interval=diagnostic_inner_interval,  # Frequency for inner optimization debug prints.
                diagnostic_outer=diagnostic_outer,  # Whether to print after each outer loop.
                residual_reduction="mean",  # How high-dimensional dynamics residual penalties are scaled.
                diagnostic_grad_norms=False,  # If True, print gradient norms for action/latent parameters.
                pad_initial_history=True,
                history_action_pad="zeros",

                dynamics_mode=dynamics_mode,
                lambda_dynamics=lambda_dynamics,
                use_dynamics_constraints=use_dynamics_constraints,

                lambda_anti_stillness = lambda_anti_stillness,
                min_action_norm = min_action_norm,

            ),
            mpc= MPCConfig(
                horizon=horizon,  # Number of macro actions planned at each MPC solve.
                execution_stride=1,  # Number of planned macro actions executed before replanning.
                warm_start=True,  # Reuse previous solution as initialization for next MPC step.
            ),
            refinement= RefinementConfig(
                enabled=False,  # If True, use extra sampling/refinement after gradient optimization.
                num_samples=0,  # Number of sampled candidate action sequences for refinement.
                noise_variance=0.3,  # Sampling noise variance for refinement.
            ),
            feasibility=FeasibilityConfig(
                enabled=feasibility_enabled,  # If True, load and use learned feasibility model.
                checkpoint_path=feasibility_checkpoint,  # Path to transformer_sigma_delta.pt or another feasibility checkpoint.

                lambda_feasibility=lambda_feasibility,  # Outer weight on total feasibility loss in solver objective.
                noise_level=0.2,  # Sigma/noise level used when evaluating DSM/noise energy.
                reduction="mean",  # Reduction style for feasibility score, usually mean over horizon.
                latent_reduction="mean",  # How to reduce predicted noise over latent dimensions, if supported by the model. "mean" or "max".
                detach_model=False,  # Must be False if feasibility should affect gradients/actions.

                # Optional:
                lambda_transition=lambda_transition,  # {22749227:1}Weight inside feasibility: action-conditioned delta transition penalty.
                lambda_action_consistency=lambda_action_consistency,  # Optional zero-action comparison term. 0 disables it.
                action_consistency_margin=0.1,  # Margin for action-consistency term if enabled.
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
    # new added 19 may
    dynamics_mode: str,
    use_dynamics_constraints: bool,
    lambda_dynamics: float ,
    lambda_anti_stillness: float,
    min_action_norm: float,
    lambda_feasibility: float,
    lambda_transition: float,
    lambda_action_consistency: float,

    episode_idx: int,
    split: str,
    horizon: int,
    feasibility_checkpoint: str, # same make model an arg
    diagnostic_inner_interval: int | None = None,
    diagnostic_outer: bool = False,
    model=None,
    model_cfg=None,
    device=None,
    debug_dynamics_action_discrimination_flag: bool = False,
    feasibility_enabled: bool = True, #also new added 19 may
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
        feasibility_checkpoint=feasibility_checkpoint,
        diagnostic_inner_interval=diagnostic_inner_interval,
        diagnostic_outer=diagnostic_outer,

        # new added 19 may
        dynamics_mode=dynamics_mode,
        use_dynamics_constraints=use_dynamics_constraints,
        lambda_dynamics=lambda_dynamics,
        lambda_anti_stillness=lambda_anti_stillness,
        min_action_norm=min_action_norm,
        feasibility_enabled=feasibility_enabled,
        lambda_feasibility=lambda_feasibility,
        lambda_transition=lambda_transition,
        lambda_action_consistency=lambda_action_consistency,
    )

    print(
        f"starting planner "
        f"(split={split}, macro_horizon={horizon}, I=25, O=25, gamma=1.9, refinement=500x0.3, "
        f"dynamics_mode={dynamics_mode}, use_dynamics_constraints={use_dynamics_constraints}, "
        f"lambda_dynamics={lambda_dynamics}, feasibility_enabled={feasibility_enabled}, "
        f"lambda_feasibility={lambda_feasibility}, lambda_transition={lambda_transition}, "
        f"lambda_action_consistency={lambda_action_consistency})"
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
            """
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
            """
    ################################################################
    # more debug
    if debug_dynamics_action_discrimination_flag:
        print("[dyn action discrim] preparing oracle latents/actions")

        with torch.no_grad():
            # Encode full oracle video plan: length horizon + 1.
            oracle_latents = world_model.encode_sequence(episode["video_plan"]).to(device)

        # The solver expects candidate_latents to include current + future latents.
        candidate_latents = oracle_latents

        # Current/past latent context.
        # For this oracle demo, we only have the start observation as context.
        # Pad by repeating/using the current latent to match history_length.
        current_latent = oracle_latents[0]
        latent_context = current_latent.unsqueeze(0).repeat(world_model.history_length, 1, 1)

        # Expert primitive actions for the whole raw horizon.
        # Shape: [horizon * FRAME_SKIP, 2]
        expert_primitive_norm = (
            episode["rel_actions"][: horizon * FRAME_SKIP].float().to(device) / 100.0
            - action_mean
        ) / action_std

        # Group 5 primitive actions into one macro action of dim 10.
        # Shape: [horizon, 10]
        expert_actions = expert_primitive_norm.reshape(horizon, wm_action_dim)

        # Past action context. Use zeros unless you have real previous actions.
        past_action_context = torch.zeros(
            world_model.history_length,
            wm_action_dim,
            device=device,
            dtype=oracle_latents.dtype,
        )

        print("[dyn action discrim] oracle_latents:", tuple(oracle_latents.shape))
        print("[dyn action discrim] latent_context:", tuple(latent_context.shape))
        print("[dyn action discrim] candidate_latents:", tuple(candidate_latents.shape))
        print("[dyn action discrim] expert_actions:", tuple(expert_actions.shape))
        print("[dyn action discrim] expert action mean/std:",
              float(expert_actions.mean().detach().cpu()),
              float(expert_actions.std().detach().cpu()))

        debug_dynamics_action_discrimination(
            planner=planner,
            latent_context=latent_context,
            candidate_latents=candidate_latents,
            expert_actions=expert_actions,
            past_action_context=past_action_context,
        )

        return {
            "episode_idx": episode_idx,
            "split": split,
            "success": False,
            "state_dist": float("nan"),
            "dynamics_residual": float("nan"),
            "executed_actions": 0,
            "planning_horizon": int(horizon),
        }

    expert_actions = expert_macro_actions(
        episode=episode,
        action_repeat=FRAME_SKIP,
        primitive_action_dim=2,
        horizon=horizon,
    )

    result = planner.run_mpc(
        observation_history=[episode["start_obs"]],
        goal_observation=episode["goal_obs"],
        step_fn=traced_step_fn,
        video_source=PrecomputedVideoPlanSource(episode["video_plan"], encoded=False),
        initial_warm_start_actions=None,
    )
    print("ALM planner finished")
    # ------------------------------------------------------------
    # Rollout objective diagnostic: expert vs planner vs zero
    # ------------------------------------------------------------
    with torch.no_grad():
        oracle_latents = world_model.encode_sequence(episode["video_plan"]).to(device)

    current_latent = oracle_latents[0]

    latent_context = current_latent.unsqueeze(0).repeat(
        world_model.history_length,
        1,
        1,
    )

    past_action_context = torch.zeros(
        world_model.history_length,
        wm_action_dim,
        device=device,
        dtype=oracle_latents.dtype,
    )

    expert_primitive_norm = (
        episode["rel_actions"][: horizon * FRAME_SKIP].float().to(device) / 100.0
        - action_mean
    ) / action_std

    expert_actions = expert_primitive_norm.reshape(horizon, wm_action_dim)

    debug_rollout_objective_action_discrimination(
        planner=planner,
        latent_context=latent_context,
        past_action_context=past_action_context,
        video_latents=oracle_latents,
        goal_latent=oracle_latents[-1],
        expert_actions=expert_actions,
        planner_actions=result.executed_actions.to(device),
    )


    first_macro = result.executed_actions[0].detach().cpu().reshape(action_repeat, primitive_action_dim)
    first_macro_raw = ((first_macro * action_std.cpu()) + action_mean.cpu()) * 100.0
    expert_rel_norm = (episode["rel_actions"][:FRAME_SKIP].float() / 100.0 - action_mean.cpu()) / action_std.cpu()

    print("first planner macro (normalized):", first_macro)
    print("first planner macro (raw env scale):", first_macro_raw)
    print("first 5 expert relative actions (normalized):", expert_rel_norm)
    print("first 5 expert relative actions (raw env scale):", episode["rel_actions"][:FRAME_SKIP])

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


# debuggin
def debug_dynamics_action_discrimination(
    planner,
    latent_context,
    candidate_latents,
    expert_actions,
    past_action_context=None,
):
    """
    Tests whether the DINO world-model dynamics residual prefers expert actions.
    Lower residual is better.
    """

    device = candidate_latents.device
    dtype = candidate_latents.dtype

    expert_actions = expert_actions.to(device=device, dtype=dtype)

    if past_action_context is None:
        past_action_context = torch.zeros(
            planner.world_model.history_length,
            expert_actions.shape[-1],
            device=device,
            dtype=dtype,
        )

    def residual_score(name, actions):
        residuals = planner.solver._dynamics_residuals(
            latent_context=latent_context,
            past_action_context=past_action_context,
            candidate_latents=candidate_latents,
            candidate_actions=actions,
        )

        mse = residuals.pow(2).mean().item()
        norm = residuals.reshape(residuals.shape[0], -1).norm(dim=1).mean().item()

        print(f"[dyn action discrim] {name:>10s} mse={mse:.6f} norm={norm:.6f}")
        return mse

    zero_actions = torch.zeros_like(expert_actions)
    neg_actions = -expert_actions

    if expert_actions.shape[0] > 1:
        perm = torch.randperm(expert_actions.shape[0], device=device)
        shuffled_actions = expert_actions[perm]
    else:
        shuffled_actions = expert_actions

    gaussian_actions = torch.randn_like(expert_actions) * expert_actions.std().clamp_min(1e-6)

    scores = {
        "expert": residual_score("expert", expert_actions),
        "zero": residual_score("zero", zero_actions),
        "shuffle": residual_score("shuffle", shuffled_actions),
        "negative": residual_score("negative", neg_actions),
        "gaussian": residual_score("gaussian", gaussian_actions),
    }

    print("[dyn action discrim] margins:")
    print(f"  zero - expert:    {scores['zero'] - scores['expert']:.6f}")
    print(f"  shuffle - expert: {scores['shuffle'] - scores['expert']:.6f}")
    print(f"  negative - expert:{scores['negative'] - scores['expert']:.6f}")
    print(f"  gaussian - expert:{scores['gaussian'] - scores['expert']:.6f}")

    return scores


@torch.no_grad()
def debug_rollout_objective_action_discrimination(
    planner,
    latent_context,
    past_action_context,
    video_latents,
    goal_latent,
    expert_actions,
    planner_actions=None,
):
    """
    Rollout objective diagnostic.

    Tests whether the actual rollout objective prefers expert actions over
    zero/shuffled/negative/gaussian/planner actions.

    Unlike debug_dynamics_action_discrimination(), this does not fix oracle
    candidate_latents. It generates candidate_latents by DINO-WM rollout:
        z_{t+1} = f_theta(h_t, a_t)
    and then evaluates the planner objective on those generated latents.
    """

    device = video_latents.device
    dtype = video_latents.dtype

    expert_actions = expert_actions.to(device=device, dtype=dtype)

    if past_action_context is None:
        past_action_context = torch.zeros(
            planner.world_model.history_length,
            expert_actions.shape[-1],
            device=device,
            dtype=dtype,
        )

    def score_actions(name, actions):
        actions = actions.to(device=device, dtype=dtype)

        rolled_latents = planner.solver._rollout_world_model(
            latent_context=latent_context,
            past_action_context=past_action_context,
            candidate_actions=actions,
        )

        objective, pieces = planner.solver._objective(
            latents=rolled_latents,
            actions=actions,
            goal_latent=goal_latent,
            video_latents=video_latents,
            action_prior=None,
        )

        feasibility_penalty, feasibility_pieces = planner.solver._feasibility_penalty(
            latent_context=latent_context,
            candidate_latents=rolled_latents,
            candidate_actions=actions,
        )

        total = objective + feasibility_penalty

        rollout_vs_video = (rolled_latents - video_latents).pow(2).mean()
        final_vs_goal = (rolled_latents[-1] - goal_latent).pow(2).mean()

        print(
            f"[rollout objective] {name:>10s} "
            f"total={float(total.detach().cpu()):.6f} "
            f"objective={float(objective.detach().cpu()):.6f} "
            f"video={float(pieces['video_loss'].detach().cpu()):.6f} "
            f"goal={float(pieces['goal_loss'].detach().cpu()):.6f} "
            f"action={float(pieces['action_loss'].detach().cpu()):.6f} "
            f"w_feas={float(feasibility_penalty.detach().cpu()):.6f} "
            f"dsm={float(feasibility_pieces.get('dsm_energy', torch.tensor(0.0)).detach().cpu()):.6f} "
            f"transition={float(feasibility_pieces.get('transition_energy', torch.tensor(0.0)).detach().cpu()):.6f} "
            f"rollout_vs_video={float(rollout_vs_video.detach().cpu()):.6f} "
            f"final_vs_goal={float(final_vs_goal.detach().cpu()):.6f}"
        )

        return float(total.detach().cpu())

    zero_actions = torch.zeros_like(expert_actions)
    neg_actions = -expert_actions

    if expert_actions.shape[0] > 1:
        perm = torch.randperm(expert_actions.shape[0], device=device)
        shuffled_actions = expert_actions[perm]
    else:
        shuffled_actions = expert_actions

    gaussian_actions = torch.randn_like(expert_actions) * expert_actions.std().clamp_min(1e-6)

    scores = {
        "expert": score_actions("expert", expert_actions),
        "zero": score_actions("zero", zero_actions),
        "shuffle": score_actions("shuffle", shuffled_actions),
        "negative": score_actions("negative", neg_actions),
        "gaussian": score_actions("gaussian", gaussian_actions),
    }

    if planner_actions is not None:
        planner_actions = planner_actions[: expert_actions.shape[0]]
        scores["planner"] = score_actions("planner", planner_actions)

    print("[rollout objective] margins total - expert:")
    for name, score in scores.items():
        if name != "expert":
            print(f"  {name:>10s} - expert: {score - scores['expert']:.6f}")

    return scores
    
def parse_args():
    parser = argparse.ArgumentParser(description="GVP-WM Oracle evaluation with ALM latent collocation")
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--horizon", type=int, default=None, help="Macro/world-model planning horizon H. For paper T=25/50/80 with frame_skip=5, use H=5/10/16.")
    parser.add_argument("--raw-horizon", type=int, choices=(25, 50, 80), default=DEFAULT_RAW_HORIZON, help="Paper/raw environment horizon T; converted to macro horizon H=T/frame_skip.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset inside the filtered eligible episode list")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of filtered episodes to evaluate")
    parser.add_argument("--debug-inner-every", type=int, default=None, help="Print ALM diagnostics every N inner iterations")
    parser.add_argument("--debug-outer", action="store_true", help="Print diagnostics after every ALM outer iteration")

    parser.add_argument("--dynamics-mode", choices=("none", "soft", "alm", "penalty", "rollout"), default="none", help="DINO-WM dynamics mode")
    parser.add_argument("--use-dynamics-constraints", type=str_to_bool, default=False, help="Whether to include dynamics as ALM constraints")
    parser.add_argument("--lambda-dynamics", type=float, default=0.0, help="weight on DINO-WM dynamics loss")
    parser.add_argument("--lambda-anti-stillness", type=float, default=0.0, help="Optional weight for anti-stillness term to encourage nonzero actions.")
    parser.add_argument("--min-action-norm", type=float, default=0.2, help="Minimum action norm for anti-stillness term (only relevant if lambda_anti_stillness > 0)")      
    parser.add_argument("--feasibility-enabled", type=str_to_bool, default=True)
    parser.add_argument("--lambda-feasibility", type=float, default=1.0, help="Outer weight on feasibility penalty in ALM objective")
    parser.add_argument("--lambda-transition", type=float, default=0.0, help="Weight on transition-consistency term inside feasibility model, if supported by the checkpoint/model")
    parser.add_argument("--lambda-action-consistency", type=float, default=0.0, help="Weight on action-consistency term inside feasibility model; 0 disables it")


    parser.add_argument("--feasibility-checkpoint",type=str,default="/home/scur0196/DL2---Grounding-Generated-Videos-/checkpoints/transformer_sigma_delta.pt",help="Path to feasibility model checkpoint")

    parser.add_argument(
        "--debug-dynamics-action-discrimination",
        action="store_true",
        help="Compare DINO-WM dynamics residual for expert, zero, shuffled, negative, and random actions, then exit.",
    )

    return parser.parse_args()



def main():
    args = parse_args()
    if args.horizon is not None:
        macro_horizon = args.horizon
        raw_horizon = macro_horizon * FRAME_SKIP
    else:
        if args.raw_horizon % FRAME_SKIP != 0:
            raise ValueError(f"--raw-horizon must be divisible by FRAME_SKIP={FRAME_SKIP}")
        raw_horizon = args.raw_horizon
        macro_horizon = raw_horizon // FRAME_SKIP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model_once(device)

    split_dir = DATA_ROOT / args.split
    eligible = candidate_episodes(split_dir, horizon=macro_horizon, frame_skip=FRAME_SKIP)
    eval_episodes = eligible[args.start_index : args.start_index + args.num_episodes]

    print(f"Evaluating split={args.split} raw_horizon={raw_horizon} macro_horizon={macro_horizon} frame_skip={FRAME_SKIP}")
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
                    horizon=macro_horizon,
                    feasibility_checkpoint=args.feasibility_checkpoint,
                    diagnostic_inner_interval=args.debug_inner_every,
                    diagnostic_outer=args.debug_outer,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                    debug_dynamics_action_discrimination_flag=args.debug_dynamics_action_discrimination,

                    # new added 19 may 
                    dynamics_mode=args.dynamics_mode,
                    use_dynamics_constraints=args.use_dynamics_constraints,
                    lambda_dynamics=args.lambda_dynamics,
                    lambda_anti_stillness=args.lambda_anti_stillness,
                    min_action_norm=args.min_action_norm,
                    feasibility_enabled=args.feasibility_enabled,
                    lambda_feasibility=args.lambda_feasibility,
                    lambda_transition=args.lambda_transition,
                    lambda_action_consistency=args.lambda_action_consistency,
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
