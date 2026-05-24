"""Parse a Plan-C single-episode log and plot residual + goal_loss vs outer.

Produces the figure attached to the email to the GVP-WM authors. Shows
how outer 1 is the sweet spot (low residual, near-zero goal loss) and
how rho saturation in subsequent outers actively pushes the iterate
off that point.

Usage:
    python scripts/plot_alm_curve.py \
        --log oracle_eval_warmstart_latents_22007234.out \
        --out alm_curve.png

The log must come from a run with diagnostic_outer=True (the
warmstart_latents / fix_adapter / full_planc scripts all set this).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

OUTER_RE = re.compile(
    r"\[alm outer (?P<outer>\d+) done\] "
    r"video=(?P<video>[\d.eE+\-nan]+) "
    r"goal=(?P<goal>[\d.eE+\-nan]+) "
    r"action=(?P<action>[\d.eE+\-nan]+) "
    r"residual=(?P<residual>[\d.eE+\-nan]+) "
    r"next_rho=(?P<next_rho>[\d.eE+\-nan]+)"
)


def parse_log(path: Path):
    outers, residuals, goals, videos, actions, next_rhos = [], [], [], [], [], []
    for line in path.read_text().splitlines():
        m = OUTER_RE.search(line)
        if not m:
            continue
        outers.append(int(m["outer"]))
        residuals.append(float(m["residual"]))
        goals.append(float(m["goal"]))
        videos.append(float(m["video"]))
        actions.append(float(m["action"]))
        next_rhos.append(float(m["next_rho"]))
    if not outers:
        raise SystemExit(
            f"No '[alm outer N done] ...' lines in {path}. "
            "Re-run with diagnostic_outer=True."
        )
    # rho used DURING outer i is the rho before this outer's dual update,
    # which equals next_rho[i-1] for i >= 1, and rho_init for i == 0.
    rhos_used = [1.0] + next_rhos[:-1]
    return outers, residuals, goals, rhos_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", default="alm_curve.png", type=Path)
    parser.add_argument(
        "--annotate-rho",
        action="store_true",
        help="annotate every outer with its rho; default annotates only key points",
    )
    args = parser.parse_args()

    outers, residuals, goals, rhos = parse_log(args.log)
    print(f"Parsed {len(outers)} outer iterations from {args.log.name}.")

    fig, ax_left = plt.subplots(figsize=(7.5, 4.5))
    ax_right = ax_left.twinx()

    color_res = "tab:blue"
    color_goal = "tab:red"

    line_res = ax_left.plot(
        outers, residuals, color=color_res, marker="o", linewidth=1.5,
        label="dynamics residual (mean per-step ‖L‖)",
    )[0]
    line_goal = ax_right.plot(
        outers, goals, color=color_goal, marker="s", linewidth=1.5,
        label="goal loss (latent MSE to encoded goal frame)",
    )[0]

    ax_left.set_xlabel("outer iteration")
    ax_left.set_ylabel("dynamics residual", color=color_res)
    ax_right.set_ylabel("goal loss", color=color_goal)
    ax_left.tick_params(axis="y", labelcolor=color_res)
    ax_right.tick_params(axis="y", labelcolor=color_goal)
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")
    ax_left.grid(True, which="both", alpha=0.3)

    # mark sweet spot (min goal_loss) and rho-saturation point
    sweet_idx = min(range(len(goals)), key=lambda i: goals[i])
    sat_idx = next(
        (i for i, r in enumerate(rhos) if r >= 1000.0 - 1e-6),
        None,
    )

    ax_right.axvline(outers[sweet_idx], color="gray", linestyle="--", alpha=0.6)
    ax_right.annotate(
        f"sweet spot\nouter {outers[sweet_idx]}\n"
        f"goal={goals[sweet_idx]:.4f}\nresidual={residuals[sweet_idx]:.1f}\n"
        f"ρ={rhos[sweet_idx]:.2f}",
        xy=(outers[sweet_idx], goals[sweet_idx]),
        xytext=(10, -50), textcoords="offset points",
        fontsize=8, color="black",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    if sat_idx is not None:
        ax_right.axvline(outers[sat_idx], color="orange", linestyle="--", alpha=0.6)
        ax_right.annotate(
            f"ρ saturates\nouter {outers[sat_idx]}\nρ=1000",
            xy=(outers[sat_idx], goals[sat_idx]),
            xytext=(10, 30), textcoords="offset points",
            fontsize=8, color="darkorange",
            bbox=dict(boxstyle="round", fc="white", ec="orange", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="orange"),
        )

    if args.annotate_rho:
        for x, g, r in zip(outers, goals, rhos):
            ax_right.annotate(f"ρ={r:.1f}", (x, g), fontsize=6,
                              xytext=(2, 4), textcoords="offset points")

    ax_left.legend(handles=[line_res, line_goal], loc="upper right", fontsize=9)
    fig.suptitle(
        "ALM optimisation trace (Plan C, ep 2, horizon 25)\n"
        f"ρ schedule: ρ₀=1, γ=1.9, ρ_max=1000  (saturates at outer "
        f"{sat_idx if sat_idx is not None else 'n/a'})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
