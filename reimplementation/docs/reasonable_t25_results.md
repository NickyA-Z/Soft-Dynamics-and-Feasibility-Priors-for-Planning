# T=25 Oracle And WAN-0S Reasonable Results Memo

Date: 2026-05-05

## Scope

This memo records the current "reasonable results" checkpoint for the Table 2
PushT/Wall, T=25 reproduction path on Snellius.

The goal of this round was not to claim full Table 2 reproduction. The goal was
to make sure the accumulated issues are separated cleanly:

- the environment, frame skip, and action scaling can produce oracle upper-bound
  sanity results;
- WAN-0S can run end to end from Wan2.1 FLF2V generation into GVP-WM evaluation;
- failures are recorded as planner/video limitations rather than hidden setup
  errors.

## Code Changes

- Added `reimplementation/scripts/reliable_t25_eval.py` for explicit reliable
  T=25 sanity policies:
  - PushT `expert`: replay dataset relative actions at the corrected scale;
  - PushT `video-track`: follow the blue agent trajectory extracted from a
    generated WAN video;
  - Wall `expert`: replay dataset actions;
  - Wall `state-track`: closed-loop track oracle states as an upper-bound sanity
    controller;
  - Wall `video-track`: track the red dot trajectory extracted from a generated
    WAN video.
- Added `reimplementation/scripts/snellius/06_reliable_t25_eval.slurm` for the
  reliable sanity evaluator.
- Updated `reimplementation/scripts/snellius/05_wan0s_t25.slurm` to accept
  `WAN0S_PROMPT` without editing the job script.
- Corrected the default PushT WAN prompt: for paper T=25 the conditioning target
  is the last frame's grey T pose, not necessarily the static green full-task
  target outline.
- Fixed PushT angle differences to use wrapped angular distance, avoiding tiny
  negative `angle_diff` values around the `2*pi` boundary.

The runtime copies of the Slurm scripts are in `~/dl2runtime/jobs/`.

## Oracle Sanity Results

PushT reliable expert replay:

- Job: `22473020`
- JSON: `~/dl2runtime/reports/reliable_pusht_expert_t25_22473020.json`
- Episodes: `0,5,6,11,13,14`
- Result: `6/6`, success rate `1.0`
- Interpretation: PushT action scale and frame skip are sane. Remaining PushT
  oracle-planner failures are inverse-action/planner-objective issues, not basic
  env replay issues.

Wall reliable state-track:

- Job: `22473003`
- JSON: `~/dl2runtime/reports/reliable_wall_state-track_t25_22473003.json`
- Episodes: `0,1,3,4,5`
- Result: `5/5`, success rate `1.0`
- Interpretation: Wall T=25 upper-bound execution is reachable when tracking
  oracle states closed-loop. Open-loop Wall expert replay was `4/5` on the same
  set because episode 4 is sensitive to replay/collision details.

## WAN-0S Results

Wall:

- Pure GVP-WM with WAN video, job `22472830`: `state_dist=5.55`, success `false`
  against the `4.5` threshold.
- WAN red-dot video-track sanity, job `22472995`: `1/1`, success rate `1.0`.
- Interpretation: Wan generates a useful Wall trajectory, but the GVP-WM action
  recovery still misses the strict success threshold on the tested Wall case.

PushT:

- Corrected-prompt WAN-0S GVP-WM batch, job `22473064`.
- JSON: `~/dl2runtime/reports/wan0s_pusht_t25_22473064.json`
- Episodes: `17,18`
- Result: `2/2`, success rate `1.0`
- Metrics:
  - ep17: `block_diff=11.96`, `angle_diff=0.158`, success `true`;
  - ep18: `block_diff=0.14`, `angle_diff=0.025`, success `true`.

Important negative controls:

- Old PushT prompt on ep13/14, job `22472996`: `1/2`; ep13 failed with
  `block_diff=124.17`, ep14 succeeded.
- Corrected PushT prompt on hard ep13, job `22473029`: still failed with
  `block_diff=72.53`, `angle_diff=1.35`.
- PushT video-track gain sweep on ep13 did not produce success; best block error
  stayed above threshold.

Interpretation: WAN-0S is now runnable and can produce successful PushT T=25
results on selected nontrivial-but-moderate episodes. It is still not robust on
harder PushT pushes/rotations, and therefore this is not yet a Table 2
reproduction.

## Artifacts

Generated videos:

- `~/dl2runtime/videos/wan0s_prompt3/pusht/episode_017/wan0s.mp4`
- `~/dl2runtime/videos/wan0s_prompt3/pusht/episode_018/wan0s.mp4`
- `~/dl2runtime/videos/wan0s/wall/episode_000/wan0s.mp4`

Contact sheets:

- `~/dl2runtime/reports/wan0s_pusht_ep017_prompt3_contact.jpg`
- `~/dl2runtime/reports/wan0s_pusht_ep018_prompt3_contact.jpg`
- `~/dl2runtime/reports/wan0s_pusht_ep013_prompt3_contact.jpg`
- `~/dl2runtime/reports/wan0s_pusht_ep014_oldprompt_contact.jpg`
- `~/dl2runtime/reports/wan0s_wall_ep000_steps8_contact.jpg`

## Current Conclusion

The project can now produce reasonable sanity results for both oracle and
WAN-0S at T=25:

- oracle/reliable upper-bound sanity: PushT `6/6`, Wall `5/5`;
- WAN-0S runnable end to end: Wall video-track `1/1`, PushT corrected-prompt
  GVP-WM `2/2` on selected moderate episodes.

The project still cannot honestly claim the paper's Table 2 PushT `0.98` and
Wall `1.00` for GVP-WM/WAN-0S. The main remaining issue is planner/action
recovery robustness: generated videos and oracle videos can be visually useful,
but the current ALM/GVP-WM inverse-action objective still collapses on harder
PushT rotations and narrowly misses Wall's threshold in pure GVP-WM mode.

## Recommended Next Step

The next reproduction step should be a fixed evaluation set with difficulty
labels:

- PushT trivial/small/moderate/hard T=25 episodes from the local 21-episode val
  set;
- Wall first 50 expert-replay-valid or state-track-valid episodes;
- paired reporting of oracle expert/state-track, WAN video-track, and pure
  GVP-WM/WAN for the same episode ids.

Only after that should we tune the planner objective further, because the
current evidence says setup-level issues are mostly under control and the
remaining gap is robustness of action recovery from video guidance.
