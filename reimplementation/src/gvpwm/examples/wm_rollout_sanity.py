"""World-model rollout sanity check.

Goal: figure out whether the latent norm explosion (~7000) we saw in
EXPERIMENT B is caused by (a) the WM checkpoint being undertrained, or
(b) our adapter feeding the WM with the wrong history / action context.

Three measurements per episode, each producing per-frame norms and
per-frame ||rollout - z_vid|| differences:

  T1. TEACHER-FORCING ONE-STEP RESIDUAL.
      For each t, take REAL latent history z_vid_{t-H+1..t} and REAL action
      history a_{t-H+1..t}, ask f to predict z_{t+1}. Compare to z_vid_{t+1}.
      If this is small -> WM itself is fine; the explosion comes from
      autoregressive compounding.

  T2. AUTOREGRESSIVE ROLLOUT WITH PADDED INIT (the way the planner uses it).
      latent_context = [z_vid_0] (padded to H by repeat_first),
      past_action_context = zeros(H-1, action_dim).
      Run adapter.rollout(...). This is exactly what experiment B did.

  T3. AUTOREGRESSIVE ROLLOUT WITH REAL HISTORY INIT.
      latent_context = z_vid_{0..H-1} (no padding),
      past_action_context = a_expert_{0..H-2} (real past actions),
      planned_actions = a_expert_{H-1..T-1}.
      If T3 << T2 -> the planner's start-of-episode padding is what blows
      the WM up; we need to fix the adapter / how planner initializes
      history. If T3 ~ T2 -> WM itself doesn't autoregress well in our setup.

Usage:
    python -m src.gvpwm.examples.wm_rollout_sanity \
        --split val --horizon 25 --episode-idx 2
"""
from __future__ import annotations

import argparse

import torch

from ..adapters.dino_wm import DinoWorldModelAdapter
from ..utils import ensure_history_length
from .dino_oracle_demo import (
    DATA_ROOT,
    FRAME_SKIP,
    DEFAULT_HORIZON,
    DEFAULT_SPLIT,
    load_model_once,
)
from .dino_oracle_diagnose import expert_macro_actions, make_world_model
from .dino_oracle_utils import load_oracle_episode, slice_oracle_episode


def per_frame_norm(z: torch.Tensor) -> torch.Tensor:
    return z.reshape(z.shape[0], -1).norm(dim=1)


def diff_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = min(a.shape[0], b.shape[0])
    return (a[:n] - b[:n]).reshape(n, -1).norm(dim=1)


def fmt(t: torch.Tensor) -> str:
    return (
        f"mean={t.mean().item():.3f}  "
        f"min={t.min().item():.3f}  "
        f"max={t.max().item():.3f}"
    )


def teacher_forcing_residuals(world_model, video_latents, expert_actions):
    H = world_model.history_length
    residuals = []
    pred_norms = []
    for t in range(H - 1, video_latents.shape[0] - 1):
        z_hist = video_latents[t - H + 1 : t + 1]
        if H > 1:
            a_hist = expert_actions[t - H + 1 : t + 1]
        else:
            a_hist = expert_actions[t : t + 1]
        z_pred = world_model.predict_next_latent(z_hist, a_hist)
        residuals.append((video_latents[t + 1] - z_pred).reshape(-1).norm())
        pred_norms.append(z_pred.reshape(-1).norm())
    return torch.stack(residuals), torch.stack(pred_norms)


def rollout_padded(world_model, video_latents, expert_actions):
    """Same init as planner: only z_vid_0, no real action history."""
    latent_context = video_latents[:1]
    latent_context = ensure_history_length(
        latent_context, world_model.history_length, pad_mode="repeat_first"
    )
    past_action_context = torch.zeros(
        max(world_model.history_length - 1, 0),
        world_model.action_dim,
        device=video_latents.device,
        dtype=video_latents.dtype,
    )
    return world_model.rollout(latent_context, past_action_context, expert_actions)


def rollout_real_history(world_model, video_latents, expert_actions):
    H = world_model.history_length
    latent_context = video_latents[:H]
    if H > 1:
        past_action_context = expert_actions[: H - 1]
    else:
        past_action_context = expert_actions[:0]
    planned = expert_actions[H - 1 :]
    return world_model.rollout(latent_context, past_action_context, planned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default=DEFAULT_SPLIT)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--episode-idx", type=int, nargs="+", default=[2])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model, model_cfg = load_model_once(device)
    world_model, primitive_action_dim, action_repeat = make_world_model(model, model_cfg, device)
    H = world_model.history_length
    print(f"world_model history_length H = {H}")
    print(f"world_model action_dim = {world_model.action_dim}")

    for episode_idx in args.episode_idx:
        print(f"\n========================= EPISODE {episode_idx} =========================")
        split_dir = DATA_ROOT / args.split
        episode = load_oracle_episode(split_dir, episode_idx)
        episode = slice_oracle_episode(episode, horizon=args.horizon, frame_skip=FRAME_SKIP)

        video_latents = world_model.encode_sequence(episode["video_plan"]).to(device)
        # video_latents has length 1 + horizon*frame_skip if dataset stores all
        # frames; the planner uses temporal_resample to horizon+1, so we mirror
        # that here so the t indexing matches expert macro actions.
        from ..video import temporal_resample_sequence
        video_latents = temporal_resample_sequence(video_latents, target_length=args.horizon + 1)
        expert = expert_macro_actions(episode, action_repeat, primitive_action_dim, args.horizon).to(device)
        print(f"video_latents shape: {tuple(video_latents.shape)}, expert shape: {tuple(expert.shape)}")

        vid_norms = per_frame_norm(video_latents)
        print(f"\n[ref] ||z_vid_t||_2 per frame: {fmt(vid_norms)}")
        print(f"[ref] ||z_vid_{{t+1}}-z_vid_t||_2 (natural step): "
              f"{fmt(diff_norm(video_latents[1:], video_latents[:-1]))}")

        # T1
        print("\n--- T1: teacher-forcing one-step prediction residuals ---")
        tf_res, tf_pred_norm = teacher_forcing_residuals(world_model, video_latents, expert)
        print(f"||z_vid_{{t+1}} - f(real_hist, real_a)||_2 over t={H - 1}..T-1: {fmt(tf_res)}")
        print(f"||predicted z||_2 (teacher forcing): {fmt(tf_pred_norm)}")

        # T2
        print("\n--- T2: autoregressive rollout with PADDED init (planner's setup) ---")
        roll2 = rollout_padded(world_model, video_latents, expert)
        print(f"rollout shape: {tuple(roll2.shape)}")
        print(f"||rollout_t||_2: {fmt(per_frame_norm(roll2))}")
        print(f"||rollout_t - z_vid_t||_2: {fmt(diff_norm(roll2, video_latents))}")

        # T3
        print("\n--- T3: autoregressive rollout with REAL history init ---")
        roll3 = rollout_real_history(world_model, video_latents, expert)
        print(f"rollout shape (starts at t=H-1): {tuple(roll3.shape)}")
        print(f"||rollout_t||_2: {fmt(per_frame_norm(roll3))}")
        # roll3 starts at t = H - 1 in absolute time
        aligned = video_latents[H - 1 : H - 1 + roll3.shape[0]]
        print(f"||rollout_t - z_vid_t||_2 (aligned at t>=H-1): {fmt(diff_norm(roll3, aligned))}")

        # Verdict
        print("\n--- verdict ---")
        tf_med = tf_res.median().item()
        nat = diff_norm(video_latents[1:], video_latents[:-1]).median().item()
        t2_max = per_frame_norm(roll2).max().item()
        t3_max = per_frame_norm(roll3).max().item()
        print(f"teacher-forcing residual median = {tf_med:.2f}, natural step median = {nat:.2f}")
        if tf_med > 5 * nat:
            print(">> teacher-forcing residual is much larger than natural step:"
                  " WM CHECKPOINT undertrained or adapter prediction wrong.")
        else:
            print(">> teacher-forcing residual is comparable to natural step: WM is fine in 1-step.")
        print(f"T2 (padded) max norm = {t2_max:.1f}, T3 (real-history) max norm = {t3_max:.1f}")
        if t2_max > 3 * t3_max:
            print(">> padded init explodes but real-history is stable:"
                  " ADAPTER / PLANNER history-padding is the bug.")
        elif t3_max > 3 * vid_norms.max().item():
            print(">> even with real history, autoregressive rollout explodes:"
                  " WM is not stable under autoregressive prediction.")
        else:
            print(">> autoregressive rollout is stable under real init.")


if __name__ == "__main__":
    main()
