"""Plot planner and oracle trajectories directly from an episode log.

Example:
    python plot.py run.out --episode 1 --output episode_1.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _points(matches: list[tuple[str, str]]) -> np.ndarray:
    return np.asarray([(float(x), float(y)) for x, y in matches], dtype=float)


def _episode_text(log_text: str, episode: int | None) -> tuple[str, int | None]:
    headings = list(re.finditer(r"^=== Episode (\d+) ===\s*$", log_text, re.MULTILINE))
    if not headings:
        if episode is not None:
            raise ValueError("The log has no '=== Episode N ===' headings")
        return log_text, None

    available = [int(match.group(1)) for match in headings]
    selected = available[0] if episode is None else episode
    if selected not in available:
        raise ValueError(f"Episode {selected} not found; available episodes: {available}")

    index = available.index(selected)
    start = headings[index].end()
    end = headings[index + 1].start() if index + 1 < len(headings) else len(log_text)
    return log_text[start:end], selected


def parse_episode_trajectories(
    log_text: str,
    episode: int | None = None,
) -> dict[str, np.ndarray | int | None]:
    """Extract macro-step positions from one episode in a planner log."""
    text, selected_episode = _episode_text(log_text, episode)

    planner_agent = _points(
        re.findall(
            rf"\[env step \d+\]\s+agent=\(({NUMBER}),({NUMBER})\)",
            text,
        )
    )
    planner_block = _points(
        re.findall(
            rf"\[env step \d+\].*?block=\(({NUMBER}),({NUMBER})\)",
            text,
        )
    )
    oracle_segments = re.findall(
        rf"oracle_agent_now=\(({NUMBER}),({NUMBER})\).*?"
        rf"oracle_agent_next=\(({NUMBER}),({NUMBER})\)",
        text,
    )
    oracle_block_logged = _points(
        re.findall(rf"oracle_t_block=\(({NUMBER}),({NUMBER})\)", text)
    )
    oracle_block_displacements = _points(
        re.findall(rf"oracle_block_disp=\(({NUMBER}),({NUMBER})\)", text)
    )

    if not len(planner_agent):
        raise ValueError("No '[env step N] agent=(x,y)' entries found")
    if not oracle_segments:
        raise ValueError("No oracle_agent_now/oracle_agent_next entries found")

    oracle_agent = np.asarray(
        [(float(oracle_segments[0][0]), float(oracle_segments[0][1]))]
        + [(float(segment[2]), float(segment[3])) for segment in oracle_segments],
        dtype=float,
    )
    if len(oracle_block_logged) and len(oracle_block_displacements):
        oracle_block = np.vstack(
            (
                oracle_block_logged[0],
                oracle_block_logged[0]
                + np.cumsum(oracle_block_displacements, axis=0),
            )
        )
    else:
        oracle_block = oracle_block_logged

    # The planner and oracle start from the same recorded state. The env-step
    # entries contain only post-action positions, so prepend that shared start.
    planner_agent = np.vstack((oracle_agent[0], planner_agent))
    if len(planner_block) and len(oracle_block):
        planner_block = np.vstack((oracle_block[0], planner_block))

    goal_match = re.search(
        rf"goal_block=\(({NUMBER}),({NUMBER}),({NUMBER})\)", text
    )
    goal_block = (
        np.asarray([float(value) for value in goal_match.groups()], dtype=float)
        if goal_match
        else np.empty((0,), dtype=float)
    )

    return {
        "episode": selected_episode,
        "planner_agent": planner_agent,
        "oracle_agent": oracle_agent,
        "planner_block": planner_block,
        "oracle_block": oracle_block,
        "goal_block": goal_block,
    }


def plot_episode_log(
    log_path: str | Path,
    *,
    episode: int | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Parse one episode from ``log_path`` and create a 2D trajectory plot."""
    log_path = Path(log_path)
    trajectories = parse_episode_trajectories(
        log_path.read_text(encoding="utf-8", errors="replace"),
        episode=episode,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    def draw(points, style, *, color, label, marker):
        if len(points):
            ax.plot(
                points[:, 0],
                points[:, 1],
                style,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                label=label,
            )

    draw(
        trajectories["planner_agent"], "-", color="tab:red",
        label="Planner agent", marker="o",
    )
    draw(
        trajectories["oracle_agent"], "-", color="tab:green",
        label="Oracle agent", marker="o",
    )
    draw(
        trajectories["planner_block"], "--", color="darkred",
        label="Planner block", marker="s",
    )
    draw(
        trajectories["oracle_block"], "--", color="limegreen",
        label="Oracle block", marker="s",
    )

    goal = trajectories["goal_block"]
    if len(goal):
        ax.scatter(
            goal[0], goal[1], marker="*", s=250, color="magenta",
            edgecolor="black", linewidth=0.5, label="Goal block", zorder=5,
        )

    # Label the shared initial position once.
    start = trajectories["planner_agent"][0]
    ax.annotate(
        "S",
        start,
        xytext=(5, -13),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

    # Index post-action planner positions consistently with [env step N].
    for step, point in enumerate(trajectories["planner_agent"][1:]):
        ax.annotate(
            f"P{step}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Oracle points after each corresponding macro action.
    for step, point in enumerate(trajectories["oracle_agent"][1:]):
        ax.annotate(
            f"O{step}",
            point,
            xytext=(5, -13),
            textcoords="offset points",
            fontsize=8,
        )

    episode_label = trajectories["episode"]
    title = "Planner vs Oracle Trajectories"
    if episode_label is not None:
        title += f" — Episode {episode_label}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    if output_path is None:
        suffix = f"_episode_{episode_label}" if episode_label is not None else ""
        output_path = log_path.with_name(f"{log_path.stem}{suffix}_trajectories.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path, trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot planner and oracle trajectories from an existing log."
    )
    parser.add_argument("log", type=Path, help="SLURM .out or text log")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    output_path, _ = plot_episode_log(
        args.log,
        episode=args.episode,
        output_path=args.output,
        show=args.show,
    )
    print(f"Saved trajectory plot: {output_path}")


if __name__ == "__main__":
    main()
