# T=25 修复真实性与论文一致性报告

日期：2026-05-05

## 结论摘要

本轮更改已经同步到远程仓库。当前需要非常明确地表述：

- 我们没有通过修改成功阈值、偷用未来 expert action、或把 diagnostic policy 冒充 GVP-WM 来“作弊”。
- 我们确实修复并验证了一批 setup-level 问题：T=25/frame skip、动作尺度、角度误差、Wan 输入输出、Slurm 运行方式、以及 prompt 与 T=25 goal 语义不一致的问题。
- 但这还不是 original paper Table 2 的完整复现。Table 2 是 50 个 initial-goal pair 的 success rate；我们目前只是完成了 sanity 和小批量 WAN-0S 成功案例。
- 当前可以对外汇报的准确说法是：代码已经具备接近论文设置的 GVP-WM/WAN-0S 端到端执行能力，并在选定 PushT/Wall T=25 sanity 上得到 reasonable results；但仍未证明能达到论文 Table 2 中 PushT 0.56 WAN-0S、Wall 0.86 WAN-0S、PushT 0.98 ORACLE、Wall 1.00 ORACLE 的 50-episode 指标。

## 和原论文 Table 2 的关系

论文 Table 2 的相关目标值是：

- PushT, GVP-WM (WAN-0S), T=25: `0.56`
- PushT, GVP-WM (ORACLE), T=25: `0.98`
- Wall, GVP-WM (WAN-0S), T=25: `0.86`
- Wall, GVP-WM (ORACLE), T=25: `1.00`

论文协议要点：

- PushT 评估 50 个 initial-goal pairs，T in `{25,50,80}`。
- Wall 评估 50 个 initial-goal pairs，T in `{25,50}`。
- goal states 只作为 visual observations 给 planner，planning 时不提供 proprioceptive goal。
- 使用 DINO-WM 预训练 action-conditioned world model，不在 evaluation 中 fine-tune world model。
- 使用 Wan2.1-FLF2V-14B 720p 作为 video generator。
- PushT 原始观测是 `224x224`，Wan 需要 `1280x720`，论文说明使用 padding/crop 做空间对齐。
- world model frame skip 是 `5`，因此 raw T=25 对应 5 个 world-model macro steps。
- GVP-WM 使用 receding-horizon execution `K=1`、ALM inner/outer `25/25`、PushT T=25 的 `lambda_r=0.05`、`lambda_v=1.0`、`lambda_g=10.0`、`rho0=1.0`、`gamma=1.9`、local refinement `500` samples with variance `0.3`；Wall T=25 使用 `gamma=1.5`。

## 目前实现中和论文一致的部分

这些部分是朝论文实现靠齐的真实修复：

- 使用现有 DINO-WM checkpoint，不 fine-tune world model。
- raw T=25 按 frame skip 5 转成 macro horizon 5。
- GVP-WM planner 使用 receding-horizon execution `K=1`。
- PushT T=25 planner hyperparameters 和论文一致：`inner=25`、`outer=25`、`lambda_action=0.05`、`lambda_video=1.0`、`lambda_goal=10.0`、`rho_growth=1.9`、refinement `500x0.3`。
- Wall T=25 使用 `rho_growth=1.5`，符合论文 appendix 的 Wall T=25 例外配置。
- Wan2.1 FLF2V 使用 first/last frame conditioning，输入被 letterbox 到 `1280x720`，输出再 center-crop 回 square frame 供 DINO-WM encoding。
- Slurm job 使用 `*.slurm`，通过 `DL2RUNTIME_ROOT`、`REIMPLEMENTATION_ROOT`、`DINO_WM_ROOT` 等环境变量配置，避免硬编码路径。

## 目前实现中不等同于论文的部分

这些地方必须诚实标注，不能拿来宣称 Table 2 复现：

- 还没有跑 50 个 fixed initial-goal pairs 的正式 evaluation，因此没有可比较的 Table 2 success rate。
- 当前 PushT WAN-0S 的正结果来自选定 episode `17,18` 的小批量 sanity，不是随机/完整 50 episode。
- Wall 的 `video-track` sanity 是从生成视频中提取红点轨迹并闭环跟踪，这是 diagnostic，不是论文 GVP-WM 方法。
- Wall 的 `state-track` oracle sanity 是用 oracle state 做闭环跟踪，也是 upper-bound diagnostic，不是论文 GVP-WM (ORACLE)。
- PushT/Wall 的 `expert` replay sanity 用数据集 action 或 state 检查环境、尺度、frame skip，这是 setup validation，不是论文 Table 2 方法。
- PushT hard case ep13 仍然失败。它说明当前 WAN-0S + GVP-WM action recovery 对更难的接触/旋转任务仍不稳。
- Wall pure GVP-WM + WAN-0S ep0 仍未过阈值：`state_dist=5.55`，而 Wall success threshold 是约 `4.5`。

## 有没有 hack 或作弊？

如果只看正式 GVP-WM/WAN-0S 评估路径，当前没有发现作弊性修改：

- 没有放宽 PushT 成功阈值。PushT 仍按 block position 和 angle 判断：`block_diff < 20` 且 `angle_diff < pi/9`。
- 没有放宽 Wall 成功阈值。Wall 仍调用环境自己的 `eval_state`。
- 没有在 WAN-0S GVP-WM 评估里执行 expert action。
- 没有在 WAN-0S GVP-WM 评估里直接跟踪 oracle state。
- 没有 fine-tune Wan 或 world model 后声称是 zero-shot。
- diagnostic policy 的输出 JSON 明确写了 policy 名，比如 `oracle_state_track`、`wan0s_red_dot_track`、`oracle_expert_replay`，没有伪装成 GVP-WM。

但是，有一个容易被误解的点：

- `wan0s_pusht_t25_22473064.json` 的 `2/2` 是 selected sanity batch，不是论文 50-episode metric。如果汇报成“PushT WAN-0S 已经 1.0”就是误导。正确说法是“WAN-0S pipeline 已能在选定非平凡 T=25 PushT episodes 上成功，但仍未完成 Table 2 统计复现”。

## 具体修复了什么

### 1. T=25 与 frame skip 修复

论文 world model frame skip 是 5，所以 raw T=25 不能当作 25 个 world-model steps；它应该对应 5 个 macro steps。当前脚本统一支持：

- `--raw-horizon 25`
- `--frame-skip 5`
- macro horizon = `raw_horizon / frame_skip = 5`

这修复了之前 planning horizon 语义容易错位的问题。

### 2. 动作尺度与 replay sanity

PushT 数据集 relative action 需要除以 `100.0` 后进入环境。Wall 环境内部把 action 乘以 2 更新 state，所以 replay dataset-scale action 时需要除以 2。我们用 `reliable_t25_eval.py` 显式验证：

- PushT expert replay job `22473020`: `6/6`
- Wall state-track job `22473003`: `5/5`

这说明环境初始化、action scale、frame skip 至少在 sanity 层面是可工作的。

### 3. 角度误差修复

原先角度差使用 `min(abs(a-b), 2*pi-abs(a-b))`，当浮点误差让 `abs(a-b)` 略大于 `2*pi` 时会出现负数。现在改成标准 wrap-around angular distance：

```python
abs((a - b + pi) % (2 * pi) - pi)
```

这不是放宽阈值，而是修复 metric 数值卫生问题。

### 4. WAN-0S 端到端 pipeline

当前 `05_wan0s_t25.slurm` 做完整流程：

1. 从 PushT/Wall episode 准备 first/last frame。
2. letterbox 到 Wan2.1 FLF2V 需要的 `1280x720`。
3. 运行 Wan2.1-FLF2V-14B 生成 `wan0s.mp4`。
4. center-crop 生成帧回 square image。
5. 送入 GVP-WM planner 作为 video guidance。
6. 执行真实环境 rollout，输出 JSON。

这个路径才是我们用来评价 GVP-WM/WAN-0S 的路径。

### 5. PushT prompt 语义修复

最重要的 WAN-0S 修复是 prompt：

旧 prompt 强调“最终与绿色目标 T 轮廓对齐”。这对 full-task goal 可能合理，但对 paper T=25 不一定合理，因为 T=25 的 goal frame 是 expert trajectory 走 25 steps 后的中间状态，不一定已经到绿色全局目标轮廓。

新 prompt 改为：

```text
绿色目标轮廓保持固定，蓝色机器人和灰色T形积木平滑移动到最后一帧所示的位置和角度。
```

这是语义修正，不是作弊：Wan 仍然只看 first/last visual frames，没有使用 expert action。这个修正让 Wan 不再被错误地拉向绿色 full-goal，而是服从最后一帧条件。

## 当前证据

Oracle/reliable sanity:

- `~/dl2runtime/reports/reliable_pusht_expert_t25_22473020.json`: PushT expert replay `6/6`
- `~/dl2runtime/reports/reliable_wall_state-track_t25_22473003.json`: Wall state-track `5/5`

WAN-0S:

- `~/dl2runtime/reports/wan0s_pusht_t25_22473064.json`: PushT corrected-prompt GVP-WM on ep17/18 `2/2`
- ep17: `block_diff=11.96`, `angle_diff=0.158`, success `true`
- ep18: `block_diff=0.14`, `angle_diff=0.025`, success `true`
- `~/dl2runtime/reports/wan0s_pusht_t25_22473029.json`: corrected-prompt ep13 still failed, `block_diff=72.53`, `angle_diff=1.35`
- `~/dl2runtime/reports/wan0s_wall_t25_22472830.json`: Wall pure GVP-WM/WAN ep0 failed narrowly, `state_dist=5.55`
- `~/dl2runtime/reports/reliable_wall_video-track_t25_22472995.json`: Wall generated video itself can be tracked successfully, `1/1`

## 推荐汇报措辞

可以这样汇报：

> 我们已经把 T=25 的运行语义、action scale、frame skip、Wan 输入输出、Slurm runtime 和 PushT prompt 语义问题修掉了。现在 oracle/reliable sanity 能证明环境和尺度没有明显错误；WAN-0S 已经能端到端跑通，并且在选定的非平凡 PushT T=25 episodes 上通过 GVP-WM 得到成功结果。不过这还不是 paper Table 2 的完整复现，因为还没有跑 50-episode official set，且 harder PushT 和 pure Wall GVP-WM 仍有失败。下一步应该固定 50-episode evaluation set，按难度分层跑 PushT/Wall，同时区分 oracle replay、video-track diagnostic 和真正的 GVP-WM/WAN-0S。

不建议这样汇报：

> 我们已经复现了 Table 2。

这句话目前不成立。

## 下一步计划

为了真正接近 paper Table 2，需要：

1. 固定 50 个 PushT T=25 initial-goal pairs 和 50 个 Wall T=25 initial-goal pairs。
2. 为每个 episode 同时记录：
   - expert/state replay sanity；
   - WAN generated video quality；
   - GVP-WM/WAN-0S rollout；
   - GVP-WM/ORACLE rollout。
3. 不再只挑 selected episodes，而是输出完整 success-rate。
4. 对失败 episode 做分类：video bad、video good but planner action recovery bad、env/action scale bad。
5. 如果多数失败是 planner action recovery bad，再继续改 ALM objective、proprio handling、video latent alignment 或 action prior。

## 同步状态

截至本报告创建前，远程 `origin/dinowm` 已同步到 commit:

```text
f39e0c6 Add reliable T25 sanity results
```
