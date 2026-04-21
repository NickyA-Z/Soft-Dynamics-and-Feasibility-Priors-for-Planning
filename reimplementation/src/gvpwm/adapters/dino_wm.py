from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from ..interfaces import WorldModelAdapter


class DinoWorldModelAdapter(WorldModelAdapter):
    def __init__(
        self,
        world_model: torch.nn.Module,
        action_dim: int,
        action_low: torch.Tensor | float = -1.0,
        action_high: torch.Tensor | float = 1.0,
        observation_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__(
            history_length=int(world_model.num_hist),
            action_dim=int(action_dim),
            action_low=action_low,
            action_high=action_high,
        )
        self.world_model = world_model
        self.observation_transform = observation_transform

    def _prepare_observation(self, observation: Any) -> Mapping[str, torch.Tensor]:
        if self.observation_transform is not None:
            observation = self.observation_transform(observation)
        if not isinstance(observation, Mapping):
            raise TypeError("DinoWorldModelAdapter expects dict-like observations.")
        visual = observation["visual"].to(self.device)
        proprio = observation.get("proprio")
        if proprio is None:
            in_chans = getattr(self.world_model.proprio_encoder, "in_chans", 0)
            proprio = torch.zeros(in_chans, device=self.device, dtype=visual.dtype)
        proprio = proprio.to(self.device)

        if visual.ndim == 3:
            visual = visual.unsqueeze(0).unsqueeze(0)
        elif visual.ndim == 4:
            visual = visual.unsqueeze(0)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0).unsqueeze(0)
        elif proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        return {"visual": visual, "proprio": proprio}

    def _pack_obs_state(self, z_obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        visual = z_obs["visual"]
        proprio = z_obs["proprio"]
        if self.world_model.concat_dim == 0:
            return torch.cat([visual, proprio.unsqueeze(2)], dim=2)
        tiled = proprio.unsqueeze(2).expand(-1, -1, visual.shape[2], -1)
        repeated = tiled.repeat(1, 1, 1, self.world_model.num_proprio_repeat)
        return torch.cat([visual, repeated], dim=3)

    def _add_action_conditioning(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        act_emb = self.world_model.encode_act(action_history.unsqueeze(0))
        if self.world_model.concat_dim == 0:
            return torch.cat([state_history.unsqueeze(0), act_emb.unsqueeze(2)], dim=2)
        tiled = act_emb.unsqueeze(2).expand(-1, -1, state_history.shape[1], -1)
        repeated = tiled.repeat(1, 1, 1, self.world_model.num_action_repeat)
        return torch.cat([state_history.unsqueeze(0), repeated], dim=3)

    def _strip_action_conditioning(self, full_state: torch.Tensor) -> torch.Tensor:
        if self.world_model.concat_dim == 0:
            return full_state[:, :, :-1, :]
        return full_state[:, :, :, :-self.world_model.action_dim]

    def encode_observation(self, observation: Any) -> torch.Tensor:
        prepared = self._prepare_observation(observation)
        with torch.no_grad():
            z_obs = self.world_model.encode_obs(prepared)
        state = self._pack_obs_state(z_obs)
        return state[0, 0]

    def encode_sequence(self, observations: Any) -> torch.Tensor:
        if isinstance(observations, Mapping):
            prepared = self._prepare_observation(observations)
            with torch.no_grad():
                z_obs = self.world_model.encode_obs(prepared)
            state = self._pack_obs_state(z_obs)
            return state[0]
        encoded = [self.encode_observation(obs) for obs in observations]
        return torch.stack(encoded, dim=0)

    def predict_next_latent(
        self,
        latent_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = self._add_action_conditioning(latent_history, action_history)
        predicted = self.world_model.predict(conditioned)
        obs_only = self._strip_action_conditioning(predicted)
        return obs_only[0, -1]

    def rollout(
        self,
        latent_context: torch.Tensor,
        past_action_context: torch.Tensor,
        planned_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        latent_context: (T, D)
        planned_actions: (H, action_dim)
        """

        latents = [latent_context[-1]]
        z_hist = latent_context.clone()
        a_hist = past_action_context.clone()

        for a in planned_actions:
            # append action
            if a_hist.numel() == 0:
                a_hist = a.unsqueeze(0)
            else:
                a_hist = torch.cat([a_hist, a.unsqueeze(0)], dim=0)

            # truncate history
            if self.history_length > 1:
                #a_hist = a_hist[-(self.history_length - 1):]
                a_hist = a_hist[-self.history_length:]

            # predict next latent
            z_next = self.predict_next_latent(z_hist, a_hist)

            # update latent history
            z_hist = torch.cat([z_hist, z_next.unsqueeze(0)], dim=0)[
                -self.history_length:
            ]

            latents.append(z_next)

        return torch.stack(latents, dim=0)
