#!/usr/bin/env python3
"""
Visualize positive and negative Push-T action trajectories used in contrastive learning.

Example:
python con_traj.py \
  --dataset /home/nvzutphen/data/feasibility_train_pusht_0_499.pt \
  --out-dir /home/nvzutphen/plots \
  --start 0 \
  --horizon 25 \
  --neg-mode wrong_action
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_actions_and_metadata(dataset_path: str | Path):
    data = torch.load(dataset_path, map_location="cpu")

    # Common formats:
    # 1. {"actions": ..., "metadata": ...}
    # 2. {"actions": ..., ...metadata keys...}
    # 3. tuple/list returned by custom code
    if isinstance(data, dict):
        if "actions" not in data:
            raise KeyError(f"Could not find key 'actions'. Available keys: {list(data.keys())}")

        actions = data["actions"]
        metadata = data.get("metadata", data)

    elif isinstance(data, (tuple, list)):
        # Try to find the tensor shaped [N, 10]
        actions = None
        metadata = {}
        for item in data:
            if torch.is_tensor(item) and item.ndim == 2 and item.shape[-1] == 10:
                actions = item
            elif isinstance(item, dict):
                metadata = item

        if actions is None:
            raise ValueError("Could not infer actions tensor from tuple/list dataset.")

    else:
        raise TypeError(f"Unsupported dataset format: {type(data)}")

    actions = actions.float().cpu()

    if actions.ndim != 2 or actions.shape[-1] != 10:
        raise ValueError(f"Expected actions shape [N, 10], got {tuple(actions.shape)}")

    return actions, metadata


def unnormalize_pusht_actions(actions: torch.Tensor, metadata: dict) -> torch.Tensor:
    """
    Input:
        actions: normalized [T, 10]

    Output:
        unnormalized [T, 10], approximately pixel/control-space actions.

    Your metadata suggests:
        a_norm = (a / action_scale - action_mean) / action_std
    so:
        a = (a_norm * action_std + action_mean) * action_scale
    """
    if "action_mean" not in metadata or "action_std" not in metadata:
        print("Warning: metadata has no action_mean/action_std. Plotting normalized actions.")
        return actions

    mean = torch.as_tensor(metadata["action_mean"], dtype=actions.dtype)
    std = torch.as_tensor(metadata["action_std"], dtype=actions.dtype)
    scale = float(metadata.get("action_scale", 100.0))

    # Usually mean/std are shape [2], because primitive actions are 2D.
    a = actions.reshape(*actions.shape[:-1], 5, 2)
    a = a * std.reshape(1, 1, 2) + mean.reshape(1, 1, 2)
    a = a * scale
    return a.reshape(*actions.shape)


def make_negative_indices(
    start: int,
    horizon: int,
    n: int,
    neg_mode: str,
    neg_offset: int,
) -> torch.Tensor:
    pos_idx = torch.arange(start, start + horizon)

    if neg_mode == "wrong_action":
        # Same idea as wrong_action contrastive: use another action sequence.
        neg_start = start + neg_offset
        if neg_start + horizon >= n:
            neg_start = max(0, start - neg_offset)

        neg_idx = torch.arange(neg_start, neg_start + horizon)

    elif neg_mode == "shuffle":
        # Random sequence of actions from elsewhere.
        g = torch.Generator().manual_seed(0)
        neg_idx = torch.randint(low=0, high=n, size=(horizon,), generator=g)

    elif neg_mode in {"small_noise", "gaussian"}:
        # In your contrastive implementation, these do NOT corrupt action.
        # They corrupt z_next only, so action trajectory is unchanged.
        neg_idx = pos_idx.clone()

    else:
        raise ValueError(f"Unknown neg_mode: {neg_mode}")

    return neg_idx


def flatten_macro_actions(actions_10d: torch.Tensor) -> torch.Tensor:
    """
    Converts [T, 10] macro-actions into [T*5, 2] primitive 2D actions.
    """
    return actions_10d.reshape(-1, 5, 2).reshape(-1, 2)


def add_arrows(ax, xy: torch.Tensor, every: int = 5):
    """
    Adds small arrows along a 2D trajectory.
    """
    arr = xy.numpy()
    for i in range(0, len(arr) - 1, every):
        x0, y0 = arr[i]
        x1, y1 = arr[i + 1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.0),
        )


def plot_positive_negative_actions(
    pos_actions: torch.Tensor,
    neg_actions: torch.Tensor,
    save_path: Path,
    title: str,
):
    pos_xy = flatten_macro_actions(pos_actions)
    neg_xy = flatten_macro_actions(neg_actions)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(pos_xy[:, 0], pos_xy[:, 1], marker="o", markersize=3, label="positive expert actions")
    ax.plot(neg_xy[:, 0], neg_xy[:, 1], marker="x", markersize=3, label="negative actions")

    add_arrows(ax, pos_xy, every=5)
    add_arrows(ax, neg_xy, every=5)

    ax.scatter(pos_xy[0, 0], pos_xy[0, 1], s=120, marker="o", label="positive start")
    ax.scatter(pos_xy[-1, 0], pos_xy[-1, 1], s=120, marker="s", label="positive end")

    ax.scatter(neg_xy[0, 0], neg_xy[0, 1], s=120, marker="x", label="negative start")
    ax.scatter(neg_xy[-1, 0], neg_xy[-1, 1], s=120, marker="D", label="negative end")

    ax.set_title(title)
    ax.set_xlabel("x action / target position")
    ax.set_ylabel("y action / target position")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_macro_panels(
    pos_actions: torch.Tensor,
    neg_actions: torch.Tensor,
    save_path: Path,
    max_panels: int = 8,
):
    """
    Small panel visualization of individual macro-actions.
    Each panel shows one 10D action as 5 primitive 2D points.
    """
    T = min(pos_actions.shape[0], max_panels)

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6), squeeze=False)

    for t in range(T):
        pos_xy = pos_actions[t].reshape(5, 2)
        neg_xy = neg_actions[t].reshape(5, 2)

        ax = axes[0, t]
        ax.plot(pos_xy[:, 0], pos_xy[:, 1], marker="o")
        add_arrows(ax, pos_xy, every=1)
        ax.set_title(f"positive a_{t}")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        ax = axes[1, t]
        ax.plot(neg_xy[:, 0], neg_xy[:, 1], marker="x")
        add_arrows(ax, neg_xy, every=1)
        ax.set_title(f"negative a_{t}")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to feasibility_train_pusht_0_499.pt")
    parser.add_argument("--out-dir", required=True, help="Directory to save figures")
    parser.add_argument("--start", type=int, default=0, help="Start index for positive sequence")
    parser.add_argument("--horizon", type=int, default=25, help="Number of macro-actions to visualize")
    parser.add_argument(
        "--neg-mode",
        choices=["wrong_action", "shuffle", "small_noise", "gaussian"],
        default="wrong_action",
        help="Negative mode matching contrastive learning",
    )
    parser.add_argument(
        "--neg-offset",
        type=int,
        default=200,
        help="Offset used for wrong_action negative sequence",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Plot normalized actions instead of unnormalized Push-T coordinates",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actions, metadata = load_actions_and_metadata(args.dataset)

    n = actions.shape[0]
    if args.start < 0 or args.start + args.horizon >= n:
        raise ValueError(
            f"Invalid start/horizon. Dataset has {n} actions, "
            f"but start={args.start}, horizon={args.horizon}."
        )

    pos_idx = torch.arange(args.start, args.start + args.horizon)
    neg_idx = make_negative_indices(
        start=args.start,
        horizon=args.horizon,
        n=n,
        neg_mode=args.neg_mode,
        neg_offset=args.neg_offset,
    )

    pos_actions = actions[pos_idx]
    neg_actions = actions[neg_idx]

    if args.neg_mode == "small_noise":
        print(
            "Note: small_noise negatives corrupt z_next, not actions. "
            "The negative action trajectory is therefore identical to the positive one."
        )
    if args.neg_mode == "gaussian":
        print(
            "Note: gaussian negatives replace z_next with noise, not actions. "
            "The negative action trajectory is therefore identical to the positive one."
        )

    if not args.normalized:
        pos_actions = unnormalize_pusht_actions(pos_actions, metadata)
        neg_actions = unnormalize_pusht_actions(neg_actions, metadata)

    main_path = out_dir / f"contrastive_actions_{args.neg_mode}_start{args.start}_H{args.horizon}.png"
    panel_path = out_dir / f"contrastive_macro_panels_{args.neg_mode}_start{args.start}_H{args.horizon}.png"

    plot_positive_negative_actions(
        pos_actions=pos_actions,
        neg_actions=neg_actions,
        save_path=main_path,
        title=f"Contrastive action trajectories: positive vs {args.neg_mode} negative",
    )

    plot_macro_panels(
        pos_actions=pos_actions,
        neg_actions=neg_actions,
        save_path=panel_path,
        max_panels=min(8, args.horizon),
    )

    print("Saved:")
    print(f"  {main_path}")
    print(f"  {panel_path}")
    print()
    print("Positive indices:", pos_idx.tolist()[:10], "...")
    print("Negative indices:", neg_idx.tolist()[:10], "...")
    print("Positive action range:", float(pos_actions.min()), float(pos_actions.max()))
    print("Negative action range:", float(neg_actions.min()), float(neg_actions.max()))


if __name__ == "__main__":
    main()