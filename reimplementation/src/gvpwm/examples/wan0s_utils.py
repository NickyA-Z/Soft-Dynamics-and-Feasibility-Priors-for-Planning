from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


WAN_SIZE = (1280, 720)
WAN_CONTENT_SIZE = 720
DEFAULT_WALL_PROMPT = (
    "俯视视角的二维几何模拟视频，白色背景，黑色墙壁和门洞保持固定，"
    "一个红色圆点从第一帧位置平滑移动到最后一帧位置。"
    "保持简洁的仿真渲染风格，不要添加文字、阴影、人物或相机运动。"
)
DEFAULT_PUSHT_PROMPT = (
    "俯视视角的二维物理模拟视频，白色背景，一个蓝色圆形机器人推动灰色T形积木，"
    "绿色目标轮廓保持固定，蓝色机器人和灰色T形积木平滑移动到最后一帧所示的位置和角度。"
    "保持简洁的仿真渲染风格，不要添加文字、阴影、人物或相机运动。"
)


def prompt_for_task(task: str) -> str:
    if task == "wall":
        return DEFAULT_WALL_PROMPT
    if task == "pusht":
        return DEFAULT_PUSHT_PROMPT
    raise ValueError(f"Unsupported WAN-0S task: {task}")


def visual_to_uint8_hwc(visual: Any) -> np.ndarray:
    tensor = torch.as_tensor(visual).detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW or HWC image, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in (1, 3):
        raise ValueError(f"Cannot infer image channels from shape={tuple(tensor.shape)}")
    if float(tensor.min()) < -0.05:
        tensor = tensor * 0.5 + 0.5
    if float(tensor.max()) <= 1.5:
        tensor = tensor * 255.0
    array = tensor.clamp(0, 255).byte().numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def letterbox_for_wan(frame: np.ndarray, size: tuple[int, int] = WAN_SIZE) -> Image.Image:
    width, height = size
    image = Image.fromarray(frame).convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    canvas = Image.new("RGB", size, (255, 255, 255))
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def center_square_to_tensor(frame: np.ndarray, task: str, image_size: int = 224) -> torch.Tensor:
    if frame.ndim != 3:
        raise ValueError(f"Expected HWC frame, got shape={frame.shape}")
    height, width = frame.shape[:2]
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    square = frame[top : top + crop, left : left + crop]
    tensor = torch.as_tensor(square, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if task in {"wall", "pusht"}:
        tensor = (tensor - 0.5) / 0.5
    return tensor


def load_wan_video_plan(video_path: str | Path, task: str, image_size: int = 224) -> list[dict[str, torch.Tensor]]:
    frames = iio.imread(video_path)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with shape THWC, got {frames.shape}")
    observations = []
    for frame in frames:
        observations.append({"visual": center_square_to_tensor(frame, task=task, image_size=image_size)})
    return observations


def write_wan0s_case(
    output_dir: str | Path,
    task: str,
    episode_idx: int,
    start_offset: int,
    start_visual: Any,
    goal_visual: Any,
    raw_horizon: int,
    frame_skip: int,
    prompt: str | None = None,
    width: int = WAN_SIZE[0],
    height: int = WAN_SIZE[1],
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    first_frame = out / "first_frame.png"
    last_frame = out / "last_frame.png"
    prompt_path = out / "prompt.txt"
    metadata_path = out / "metadata.json"

    wan_size = (int(width), int(height))
    letterbox_for_wan(visual_to_uint8_hwc(start_visual), size=wan_size).save(first_frame)
    letterbox_for_wan(visual_to_uint8_hwc(goal_visual), size=wan_size).save(last_frame)
    prompt = prompt or prompt_for_task(task)
    prompt_path.write_text(prompt, encoding="utf-8")

    metadata = {
        "task": task,
        "episode_idx": int(episode_idx),
        "start_offset": int(start_offset),
        "raw_horizon": int(raw_horizon),
        "frame_skip": int(frame_skip),
        "width": int(width),
        "height": int(height),
        "first_frame": str(first_frame),
        "last_frame": str(last_frame),
        "prompt": prompt,
        "expected_video": str(out / "wan0s.mp4"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
