from __future__ import annotations

import torch

from .config import ALMConfig, FeasibilityConfig, SolverConfig
from .feasibility_model.model import FeasibilityModel
from .interfaces import CollocationResult, WorldModelAdapter
from .losses import goal_mse, scale_invariant_alignment, squared_norm
from .utils import ensure_history_length


class LatentCollocationSolver:
    def __init__(
        self,
        world_model: WorldModelAdapter,
        config: SolverConfig,
        config_alm: ALMConfig,
        config_feasibility: FeasibilityConfig,
        feasibility_model: FeasibilityModel | None = None,
    ) -> None:
        if config_feasibility.enabled and feasibility_model is None:
            raise ValueError("Feasibility model must be provided if feasibility is enabled.")

        self.world_model = world_model
        self.config = config
        self.config_alm = config_alm
        self.config_feasibility = config_feasibility
        self.feasibility_model = feasibility_model

    def _move_to_device(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.world_model.device
        return (
            latent_context.to(device),
            past_action_context.to(device),
            goal_latent.to(device),
            video_latents.to(device),
        )

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

    def _init_latents(
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
            latents = video_latents.clone()
        else:
            view_shape = (horizon + 1,) + (1,) * current_latent.ndim
            alpha = torch.linspace(
                0.0,
                1.0,
                horizon + 1,
                device=current_latent.device,
                dtype=current_latent.dtype,
            ).view(view_shape)
            latents = current_latent.unsqueeze(0) + alpha * (goal_latent.unsqueeze(0) - current_latent.unsqueeze(0))
        latents[0] = current_latent
        return latents

    def _init_actions(
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

    def _init_params(
        self,
        initial_latents: torch.Tensor,
        horizon: int,
        warm_start_actions: torch.Tensor | None,
    ) -> tuple[torch.nn.Parameter | None, torch.nn.Parameter, list[torch.nn.Parameter]]:
        latent_parameter = None if self.config.fix_states_to_video else torch.nn.Parameter(initial_latents[1:].clone())
        action_parameter = torch.nn.Parameter(
            self._init_actions(
                horizon=horizon,
                warm_start_actions=warm_start_actions,
                device=self.world_model.device,
            )
        )
        parameters = [action_parameter]
        if latent_parameter is not None:
            parameters.insert(0, latent_parameter)
        return latent_parameter, action_parameter, parameters

    def _init_alm_state(
        self,
        horizon: int,
        current_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        residual_shape = (horizon,) + tuple(current_latent.shape)
        multipliers = torch.zeros(
            residual_shape,
            device=self.world_model.device,
            dtype=current_latent.dtype,
        )
        rho = float(self.config_alm.rho_init)
        return multipliers, rho

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
                video_loss = video_loss + scale_invariant_alignment(
                    latents[index],
                    video_latents[index],
                )
        goal_loss = goal_mse(latents[-1], goal_latent)
        action_loss = actions.pow(2).sum()

        objective = self.config.lambda_video * video_loss
        objective += self.config.lambda_goal * goal_loss
        objective += self.config.lambda_action * action_loss

        return objective, {
            "video_loss": video_loss,
            "goal_loss": goal_loss,
            "action_loss": action_loss,
        }

    def _build_candidate_latents(
        self,
        current_latent: torch.Tensor,
        latent_parameter: torch.nn.Parameter | None,
        initial_latents: torch.Tensor,
    ) -> torch.Tensor:
        if latent_parameter is None:
            return initial_latents
        return torch.cat([current_latent.unsqueeze(0), latent_parameter], dim=0)

    def _finalize(
        self,
        current_latent: torch.Tensor,
        latent_parameter: torch.nn.Parameter | None,
        action_parameter: torch.nn.Parameter,
        initial_latents: torch.Tensor,
        final_objective: torch.Tensor,
        final_augmented: torch.Tensor,
        final_residuals: torch.Tensor,
        multipliers: torch.Tensor,
        rho: float,
        diagnostics: dict[str, float],
    ) -> CollocationResult:
        final_actions = self._actions_from_parameter(action_parameter).detach()
        if latent_parameter is None:
            final_latents = initial_latents.detach()
        else:
            final_latents = torch.cat(
                [current_latent.unsqueeze(0), latent_parameter.detach()],
                dim=0,
            )
        residual_norm = final_residuals.reshape(final_residuals.shape[0], -1).norm(dim=1).mean()
        diagnostics.update({"rho": float(rho), "residual_norm": float(residual_norm.cpu())})
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

    def _solve_alm(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        current_latent: torch.Tensor,
        initial_latents: torch.Tensor,
        latent_parameter: torch.nn.Parameter | None,
        action_parameter: torch.nn.Parameter,
        parameters: list[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        multipliers: torch.Tensor,
        rho: float,
    ) -> CollocationResult:
        diagnostics: dict[str, float] = {}
        final_objective = current_latent.new_tensor(0.0)
        final_augmented = current_latent.new_tensor(0.0)
        final_residuals = torch.zeros_like(multipliers)

        for _ in range(self.config_alm.outer_steps):
            for _ in range(self.config.inner_steps):
                optimizer.zero_grad()
                actions = self._actions_from_parameter(action_parameter)
                candidate_latents = self._build_candidate_latents(current_latent, latent_parameter, initial_latents)
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
                for index in range(residuals.shape[0]):
                    augmented = augmented + (multipliers[index] * residuals[index]).sum()
                    augmented = augmented + 0.5 * rho * squared_norm(residuals[index])
                augmented.backward()
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
                diagnostics = {key: float(value.detach().cpu()) for key, value in pieces.items()}

            with torch.no_grad():
                multipliers = multipliers + rho * final_residuals
                rho = min(rho * self.config_alm.rho_growth, self.config_alm.rho_max)

        return self._finalize(
            current_latent=current_latent,
            latent_parameter=latent_parameter,
            action_parameter=action_parameter,
            initial_latents=initial_latents,
            final_objective=final_objective,
            final_augmented=final_augmented,
            final_residuals=final_residuals,
            multipliers=multipliers,
            rho=rho,
            diagnostics=diagnostics,
        )

    def _solve_feasibility(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        current_latent: torch.Tensor,
        initial_latents: torch.Tensor,
        latent_parameter: torch.nn.Parameter | None,
        action_parameter: torch.nn.Parameter,
        parameters: list[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        multipliers: torch.Tensor,
        rho: float,
    ) -> CollocationResult:
        raise NotImplementedError

    def _solve_langevin_alm(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        current_latent: torch.Tensor,
        initial_latents: torch.Tensor,
        latent_parameter: torch.nn.Parameter | None,
        action_parameter: torch.nn.Parameter,
        parameters: list[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        multipliers: torch.Tensor,
        rho: float,
    ) -> CollocationResult:
        raise NotImplementedError

    def solve(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        goal_latent: torch.Tensor,
        video_latents: torch.Tensor,
        warm_start_latents: torch.Tensor | None = None,
        warm_start_actions: torch.Tensor | None = None,
    ) -> CollocationResult:
        latent_context, past_action_context, goal_latent, video_latents = self._move_to_device(
            latent_context, past_action_context, goal_latent, video_latents
        )

        current_latent = latent_context[-1]
        horizon = video_latents.shape[0] - 1

        initial_latents = self._init_latents(current_latent, goal_latent, video_latents, warm_start_latents)
        latent_param, action_param, params = self._init_params(initial_latents, horizon, warm_start_actions)
        multipliers, rho = self._init_alm_state(horizon, current_latent)
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate, eps=self.config.adam_eps)

        shared_kwargs = dict(
            latent_context=latent_context,
            past_action_context=past_action_context,
            goal_latent=goal_latent,
            video_latents=video_latents,
            current_latent=current_latent,
            initial_latents=initial_latents,
            latent_parameter=latent_param,
            action_parameter=action_param,
            parameters=params,
            optimizer=optimizer,
            multipliers=multipliers,
            rho=rho,
        )

        if self.config_alm.enabled and self.config_feasibility.enabled:
            return self._solve_langevin_alm(**shared_kwargs)
        if self.config_alm.enabled:
            return self._solve_alm(**shared_kwargs)
        if self.config_feasibility.enabled:
            return self._solve_feasibility(**shared_kwargs)
        raise ValueError("At least one of ALM or feasibility configs must be enabled.")
