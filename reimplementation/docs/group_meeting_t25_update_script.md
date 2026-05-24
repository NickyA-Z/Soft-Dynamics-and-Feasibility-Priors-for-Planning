# Group Meeting Script: T=25 Oracle and WAN-0S Reproduction Update

## Opening

Hi everyone, today I want to give a short update on the T=25 reproduction work for the PushT and Wall experiments from Table 2 of the GVP-WM paper.

The main message is: we have now fixed several setup-level issues and can run the WAN-0S pipeline end to end on Snellius, but we should not yet claim that we have reproduced Table 2. What we have is a much cleaner and more honest checkpoint: oracle sanity checks are now reasonable, WAN-0S is runnable, and we have successful selected PushT T=25 cases. However, we still need a full 50-episode evaluation before comparing directly to the paper numbers.

## What We Were Trying To Reproduce

The target experiment is Table 2, specifically PushT and Wall with GVP-WM, WAN-0S, and T=25.

From the paper, the relevant reported success rates are:

- PushT, GVP-WM with WAN-0S, T=25: 0.56.
- PushT, GVP-WM with ORACLE video, T=25: 0.98.
- Wall, GVP-WM with WAN-0S, T=25: 0.86.
- Wall, GVP-WM with ORACLE video, T=25: 1.00.

The important detail is that these are success rates over 50 initial-goal pairs. So our selected sanity results are useful evidence that the implementation is moving in the right direction, but they are not yet the paper metric.

## What We Fixed

There were several issues that could easily make the experiment look broken for the wrong reasons.

First, we fixed the T=25 interpretation. The DINO-WM world model uses frame skip 5, so raw T=25 corresponds to 5 world-model macro steps, not 25 macro steps. The scripts now explicitly use `raw_horizon=25` and `frame_skip=5`.

Second, we validated action scaling. For PushT, dataset relative actions need to be divided by 100 before execution in the environment. For Wall, the environment internally applies a factor of 2 to actions, so replaying dataset-scale Wall actions requires dividing by 2. We added a reliable sanity evaluator to make these checks explicit.

Third, we fixed the angle metric. The previous angular difference computation could produce tiny negative values around the 2-pi boundary due to floating point wrap-around. This did not intentionally change the success threshold; it just made the metric mathematically correct.

Fourth, we made WAN-0S run end to end on Snellius. The job now prepares first and last frames, letterboxes them for Wan2.1 FLF2V, generates the video, crops the generated frames back to square observations, feeds them into GVP-WM, and writes JSON metrics.

Fifth, we fixed the PushT prompt semantics. The old prompt encouraged the model to align the grey T block with the green target outline. That is not always correct for T=25, because the final frame is often an intermediate expert state, not the full task goal. The prompt now says that the green outline should stay fixed, and that the blue agent and grey T should move to the position and angle shown in the last frame.

## Integrity Check

I also want to be very clear about what this is and is not.

We did not relax the success thresholds. PushT still requires the final block pose to satisfy both the position and rotation thresholds. Wall still uses the environment's own success metric.

We did not use expert actions inside the WAN-0S GVP-WM evaluation. We also did not fine-tune Wan or the world model and call it zero-shot.

Some of the new evaluators are diagnostic tools, not paper methods. For example, Wall state tracking and red-dot video tracking are sanity checks to test whether the environment and generated video trajectory are reasonable. They are explicitly labeled as diagnostic policies in the JSON outputs, and we should not present them as GVP-WM results.

So the honest statement is: the implementation is cleaner and closer to the paper setup, but we have not yet reproduced the paper's 50-episode Table 2 numbers.

## Results So Far

For oracle sanity, the results are now reasonable.

PushT expert replay succeeded on 6 out of 6 selected T=25 episodes. This tells us that the PushT action scale and frame skip are correct at the environment level.

Wall state-track sanity succeeded on 5 out of 5 selected episodes. This tells us that the Wall T=25 target states are reachable when tracking oracle states closed-loop.

For WAN-0S, we now have successful end-to-end PushT cases with the corrected prompt.

On selected PushT episodes 17 and 18, GVP-WM with WAN-0S succeeded on both cases. Episode 17 is especially useful because it has nonzero block motion: the final metrics were approximately block error 11.96 and angle error 0.158, both within the success thresholds.

We also kept negative controls. Harder PushT episode 13 still fails, even with the corrected prompt. Wall pure GVP-WM with WAN-0S on episode 0 also still misses the success threshold narrowly. So the remaining issue is not just setup. It appears to be robustness of action recovery from video guidance, especially on harder contact-rich PushT rotations and on strict Wall thresholds.

## How Close Are We To The Paper?

Conceptually, several core pieces now match the paper: DINO-WM checkpoints, frame skip 5, Wan2.1 FLF2V first-last-frame generation, spatial padding and cropping, receding-horizon execution with K=1, and the main ALM hyperparameters for T=25.

But experimentally, we are not yet at the paper protocol. The paper reports success rate over 50 initial-goal pairs. We have selected sanity batches and diagnostic checks. That is useful for debugging, but it is not a replacement for the official evaluation.

So I would describe the status as: setup-level blockers are mostly cleared, WAN-0S is operational, and selected cases now work; the next bottleneck is scaling this to the full evaluation set and improving robustness on hard failures.

## Proposed Next Step

The next step should be a fixed 50-episode evaluation for PushT T=25 and Wall T=25.

For each episode, I think we should log three things side by side:

1. Oracle or expert replay sanity, to check whether the target is reachable under the environment setup.
2. Generated video quality, ideally with a lightweight diagnostic like object tracking or contact-sheet inspection.
3. The actual GVP-WM rollout result using the generated WAN-0S video.

This will let us classify failures into three buckets: bad generated video, good video but bad action recovery, or remaining environment/setup mismatch.

Only after that should we tune the planner further, because now the evidence suggests the remaining gap is mostly planner and action-recovery robustness rather than basic Slurm or data setup.

## Closing

To summarize: we should not claim Table 2 reproduction yet. But this is meaningful progress. We now have a working Snellius WAN-0S pipeline, corrected T=25 semantics, validated action scales, and selected successful nontrivial PushT WAN-0S cases. The next milestone is to run the full 50-episode evaluation and compare honestly against the paper's reported success rates.
