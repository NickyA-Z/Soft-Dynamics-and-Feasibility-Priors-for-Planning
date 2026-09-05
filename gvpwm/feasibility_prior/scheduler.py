from abc import ABC, abstractmethod
import torch


class SigmaScheduler(ABC):
    @abstractmethod
    def sample_sigmas(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        raise NotImplementedError

    def add_noise(self, x: torch.Tensor, sigma: torch.Tensor):
        eps = torch.randn_like(x)
        noisy = x + sigma.reshape(-1, *([1] * (x.ndim - 1))) * eps
        return noisy, eps

    @abstractmethod
    def loss_weight(self, sigma: torch.Tensor):
        """Training weighting (larger sigmas might dominate)."""
        raise NotImplementedError

    def energy_weight(
        self,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Inference normalization."""
        sigma = sigma.clamp(min=1e-6)
        return 1.0 / sigma.pow(2)


class LogUniformSigmaScheduler(SigmaScheduler):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample_sigmas(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        u = torch.rand(batch_size, 1, device=device, dtype=dtype)
        sigma_min = torch.as_tensor(self.sigma_min, device=device, dtype=dtype)
        sigma_max = torch.as_tensor(self.sigma_max, device=device, dtype=dtype)
        log_sigma = torch.log(sigma_min) + u * (torch.log(sigma_max) - torch.log(sigma_min))
        return torch.exp(log_sigma)

    def loss_weight(self, sigma: torch.Tensor):
        sigma = sigma.clamp(min=1e-6)
        return torch.ones_like(sigma)
        # return 1.0 / sigma
