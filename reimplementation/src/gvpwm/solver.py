from __future__ import annotations

import torch

from .config import ALMConfig
from .interfaces import CollocationResult, WorldModelAdapter
from .losses import squared_norm
from .utils import ensure_history_length


class LatentCollocationSolver:
    def __init__(self, world_model: WorldModelAdapter, config: ALMConfig) -> None:
        self.world_model = world_model
        self.config = config

    def _actions_from_parameter(self, action_parameter: torch.Tensor) -> torch.Tensor:
        if not self.config.use_action_reparameterization:
            return torch.clamp(
                action_parameter,
                min=self.world_model.action_low,
                max=self.world_model.action_high,
            )
        low = self.world_model.action_low.to(action_parameter)
        high = self.world_model.action_high.to(action_parameter)
        scaled = torch.tanh(action_parameter)
        return low + 0.5 * (scaled + 1.0) * (high - low)

    def _raw_parameter_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.config.use_action_reparameterization:
            return actions.clone()
        low = self.world_model.action_low.to(actions)
        high = self.world_model.action_high.to(actions)
        scaled = 2.0 * (actions - low) / (high - low).clamp_min(1e-8) - 1.0
        scaled = scaled.clamp(-0.999999, 0.999999)
        return torch.atanh(scaled)

    def _initialize_latents(
        self,
        current_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        warm_start_latents: torch.Tensor | None,
    ) -> torch.Tensor:
        horizon = video_latents.shape[0] - 1
        if warm_start_latents is not None and warm_start_latents.shape[0] == horizon + 1:
            latents = warm_start_latents.clone()
        elif self.config.use_video_init:
            latents = self.world_model.initialize_latents_from_video(
                current_latent=current_latent,
                video_latents=video_latents,
            )
        else:
            view_shape = (horizon + 1,) + (1,) * current_latent.ndim
            alpha = torch.linspace(
                0.0,
                1.0,
                horizon + 1,
                device=current_latent.device,
                dtype=current_latent.dtype,
            ).view(view_shape)
            latents = current_latent.unsqueeze(0) + alpha * (
                goal_latent.unsqueeze(0) - current_latent.unsqueeze(0)
            )
        latents[0] = current_latent
        return latents

    def _initialize_actions(
        self,
        horizon: int,
        warm_start_actions: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if warm_start_actions is not None and warm_start_actions.shape[0] == horizon:
            return self._raw_parameter_from_actions(warm_start_actions.to(device))
        return torch.zeros(
            horizon,
            self.world_model.action_dim,
            device=device,
            dtype=self.world_model.action_low.dtype,
        )

    def _dynamics_residuals(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        candidate_latents: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        history = ensure_history_length(
            latent_context,
            self.world_model.history_length,
            pad_mode="repeat_first",
        )
        if self.world_model.history_length > 1:
            action_context = ensure_history_length(
                past_action_context,
                self.world_model.history_length - 1,
                pad_mode="zeros",
            )
        else:
            action_context = candidate_actions.new_zeros((0, self.world_model.action_dim))

        residuals = []
        for index in range(candidate_actions.shape[0]):
            state_window = torch.cat([history, candidate_latents[1 : index + 1]], dim=0)[
                -self.world_model.history_length :
            ]
            action_window = torch.cat(
                [action_context, candidate_actions[: index + 1]],
                dim=0,
            )[-self.world_model.history_length :]
            predicted_next = self.world_model.predict_next_latent(state_window, action_window)
            residuals.append(candidate_latents[index + 1] - predicted_next)
        return torch.stack(residuals, dim=0)

    def _objective(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        video_loss = latents.new_tensor(0.0)
        if self.config.use_video_loss and latents.shape[0] > 2:
            for index in range(1, latents.shape[0] - 1):
                video_loss = video_loss + self.world_model.video_alignment_loss(
                    latents[index],
                    video_latents[index],
                )
        goal_loss = self.world_model.goal_loss(latents[-1], goal_latent)
        action_loss = actions.pow(2).sum()
        objective = (
            self.config.lambda_video * video_loss
            + self.config.lambda_goal * goal_loss
            + self.config.lambda_action * action_loss
        )
        return objective, {
            "video_loss": video_loss,
            "goal_loss": goal_loss,
            "action_loss": action_loss,
        }

    def solve(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        warm_start_latents: torch.Tensor | None = None,
        warm_start_actions: torch.Tensor | None = None,
    ) -> CollocationResult:
        latent_context = latent_context.to(self.world_model.device)
        past_action_context = past_action_context.to(self.world_model.device)
        goal_latent = goal_latent.to(self.world_model.device)
        video_latents = video_latents.to(self.world_model.device)

        current_latent = latent_context[-1]
        horizon = video_latents.shape[0] - 1
        initial_latents = self._initialize_latents(
            current_latent=current_latent,
            goal_latent=goal_latent,
            video_latents=video_latents,
            warm_start_latents=warm_start_latents,
        )

        if self.config.fix_states_to_video:
            latent_parameter = None
        else:
            latent_parameter = torch.nn.Parameter(initial_latents[1:].clone())

        action_parameter = torch.nn.Parameter(
            self._initialize_actions(
                horizon=horizon,
                warm_start_actions=warm_start_actions,
                device=self.world_model.device,
            )
        )

        parameters = [action_parameter]
        if latent_parameter is not None:
            parameters.insert(0, latent_parameter)
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_eps,
        )

        residual_shape = (horizon,) + tuple(current_latent.shape)
        multipliers = torch.zeros(
            residual_shape,
            device=self.world_model.device,
            dtype=current_latent.dtype,
        )
        rho = float(self.config.rho_init)

        diagnostics: dict[str, float] = {}
        final_objective = current_latent.new_tensor(0.0)
        final_augmented = current_latent.new_tensor(0.0)
        final_residuals = torch.zeros_like(multipliers)

        for outer_index in range(self.config.outer_steps):
            for inner_index in range(self.config.inner_steps):
                optimizer.zero_grad()
                actions = self._actions_from_parameter(action_parameter)
                if latent_parameter is None:
                    candidate_latents = initial_latents
                else:
                    candidate_latents = torch.cat(
                        [current_latent.unsqueeze(0), latent_parameter],
                        dim=0,
                    )
                residuals = self._dynamics_residuals(
                    latent_context=latent_context,
                    past_action_context=past_action_context,
                    candidate_latents=candidate_latents,
                    candidate_actions=actions,
                )
                objective, pieces = self._objective(
                    latents=candidate_latents,
                    actions=actions,
                    goal_latent=goal_latent,
                    video_latents=video_latents,
                )
                augmented = objective
                if self.config.residual_reduction == "mean":
                    pen_scale = 1.0 / float(residuals[0].numel())
                else:
                    pen_scale = 1.0
                for index in range(residuals.shape[0]):
                    augmented = augmented + (multipliers[index] * residuals[index]).sum()
                    augmented = augmented + 0.5 * rho * pen_scale * squared_norm(residuals[index])
                augmented.backward()
                if (
                    self.config.diagnostic_grad_norms
                    and self.config.diagnostic_inner_interval is not None
                    and (
                        inner_index % self.config.diagnostic_inner_interval == 0
                        or inner_index == self.config.inner_steps - 1
                    )
                ):
                    a_grad = action_parameter.grad
                    a_norm = float(a_grad.norm().cpu()) if a_grad is not None else float("nan")
                    if latent_parameter is not None and latent_parameter.grad is not None:
                        l_grad = latent_parameter.grad
                        l_norm = float(l_grad.norm().cpu())
                        l_mean_abs = float(l_grad.abs().mean().cpu())
                    else:
                        l_norm = float("nan")
                        l_mean_abs = float("nan")
                    print(
                        f"[grad outer {outer_index} inner {inner_index}] "
                        f"action_grad_norm={a_norm:.4e} "
                        f"latent_grad_norm={l_norm:.4e} "
                        f"latent_grad_mean_abs={l_mean_abs:.4e}"
                    )
                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.config.clip_grad_norm)
                optimizer.step()
                if not self.config.use_action_reparameterization:
                    with torch.no_grad():
                        action_parameter.clamp_(
                            min=self.world_model.action_low,
                            max=self.world_model.action_high,
                        )
                final_objective = objective.detach()
                final_augmented = augmented.detach()
                final_residuals = residuals.detach()
                diagnostics = {
                    key: float(value.detach().cpu())
                    for key, value in pieces.items()
                }
                if (
                    self.config.diagnostic_inner_interval is not None
                    and (
                        inner_index % self.config.diagnostic_inner_interval == 0
                        or inner_index == self.config.inner_steps - 1
                    )
                ):
                    inner_residual_norm = residuals.reshape(residuals.shape[0], -1).norm(dim=1).mean()
                    print(
                        f"[alm outer {outer_index} inner {inner_index}] "
                        f"video={diagnostics['video_loss']:.6f} "
                        f"goal={diagnostics['goal_loss']:.6f} "
                        f"action={diagnostics['action_loss']:.6f} "
                        f"residual={float(inner_residual_norm.cpu()):.6f} "
                        f"rho={rho:.6f}"
                    )

            with torch.no_grad():
                if self.config.residual_reduction == "mean":
                    dual_scale = 1.0 / float(final_residuals[0].numel())
                else:
                    dual_scale = 1.0
                multipliers = multipliers + rho * dual_scale * final_residuals
                rho = min(rho * self.config.rho_growth, self.config.rho_max)
                if self.config.diagnostic_outer:
                    outer_residual_norm = final_residuals.reshape(final_residuals.shape[0], -1).norm(dim=1).mean()
                    print(
                        f"[alm outer {outer_index} done] "
                        f"video={diagnostics['video_loss']:.6f} "
                        f"goal={diagnostics['goal_loss']:.6f} "
                        f"action={diagnostics['action_loss']:.6f} "
                        f"residual={float(outer_residual_norm.cpu()):.6f} "
                        f"next_rho={rho:.6f}"
                    )

        final_actions = self._actions_from_parameter(action_parameter).detach()
        if latent_parameter is None:
            final_latents = initial_latents.detach()
        else:
            final_latents = torch.cat(
                [current_latent.unsqueeze(0), latent_parameter.detach()],
                dim=0,
            )
        residual_norm = final_residuals.reshape(final_residuals.shape[0], -1).norm(dim=1).mean()
        diagnostics.update(
            {
                "rho": float(rho),
                "residual_norm": float(residual_norm.cpu()),
            }
        )
        return CollocationResult(
            latents=final_latents,
            actions=final_actions,
            objective=float(final_objective.cpu()),
            augmented_lagrangian=float(final_augmented.cpu()),
            dynamics_residual_norm=float(residual_norm.cpu()),
            multipliers=multipliers.detach(),
            rho=rho,
            diagnostics=diagnostics,
        )
