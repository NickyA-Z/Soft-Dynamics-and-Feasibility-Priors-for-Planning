# WAN-0S Implementation Memo

Date: 2026-05-05

## Working Definition

WAN-0S means zero-shot Wan video guidance:

- use the off-the-shelf Wan2.1 FLF2V checkpoint already staged in `~/dl2runtime`;
- do not fine-tune Wan and do not train task-specific LoRA;
- condition Wan on the environment start frame and goal frame;
- feed the generated video frames into the GVP-WM planner as the video plan.

This intentionally ignores the current oracle-planner accuracy issue. The immediate goal is to make the WAN-0S pipeline executable end to end on Snellius and produce inspectable generated videos plus planner metrics.

## Pipeline

1. Prepare FLF2V inputs from an existing PushT or Wall episode.
2. Letterbox the square environment frames into Wan's supported `1280*720` canvas.
3. Run `generate.py --task flf2v-14B` from the local Wan2.1 checkout.
4. Read the generated mp4, center-crop the 720px square content, resize to 224, and use visual-only observations as the GVP-WM video plan.
5. Run the existing GVP-WM MPC loop against the real PushT/Wall environment.
6. Save generated videos, planner JSON, and a run report.

## Practical Choices

- Default task for first runnable result: Wall, episode 0, raw T=25, frame skip 5.
- Wan frame count: 81 frames by default. This local Wan2.1 FLF2V implementation hard-codes an 81-frame mask, so shorter `frame_num` values fail before sampling.
- Prompt language: Chinese, because Wan FLF2V recommends Chinese prompts.
- Generated frames are visual-only; proprio is omitted so the DINO adapter uses its default zero-proprio behavior for video guidance while goal observations keep true proprio.

## Success Criteria For This Step

The pipeline is considered runnable when:

- the Slurm job finishes successfully;
- a non-empty Wan mp4 is produced;
- GVP-WM evaluation consumes that mp4 without crashing;
- the report records at least one planner result with success flag, distance metric, and output path.

The result does not need to match Table 2 yet.
