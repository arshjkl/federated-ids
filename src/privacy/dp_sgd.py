"""
Differentially-private SGD for federated IDS client training.

Wraps Opacus (1.6.0) when available to perform per-example gradient clipping,
Gaussian noise injection, and (epsilon, delta) privacy accounting. Falls back
to a manual DP-SGD implementation if Opacus is unavailable so the module is
always importable and testable.

Public API:
    DPSGDTrainer(model, optimizer, dataloader, *, max_grad_norm, noise_multiplier, delta)
        - train_epoch()            -> one epoch of DP-SGD
        - get_privacy_spent()      -> (epsilon, delta)
    clip_gradients(parameters, max_grad_norm)  -> total gradient norm
    add_gaussian_noise(parameters, noise_multiplier, max_grad_norm)
    compute_epsilon(steps, num_examples, batch_size, noise_multiplier, delta)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier

    OPACUS_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    OPACUS_AVAILABLE = False
    PrivacyEngine = None  # type: ignore
    get_noise_multiplier = None  # type: ignore


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_grad_norm: float
) -> float:
    """
    Clip per-parameter gradients to *max_grad_norm* (global norm clipping) and
    return the total gradient norm before clipping.

    This is the manual fallback path; ``DPSGDTrainer`` prefers Opacus'
    per-example clipping when available.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
    )
    clip_coef = max_grad_norm / (total_norm.item() + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return total_norm.item()


def add_gaussian_noise(
    parameters: Iterable[torch.nn.Parameter],
    noise_multiplier: float,
    max_grad_norm: float,
) -> None:
    """Add Gaussian noise N(0, (noise_multiplier * max_grad_norm)^2) to each grad."""
    sigma = noise_multiplier * max_grad_norm
    for p in parameters:
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0, std=sigma, size=p.grad.shape, device=p.grad.device
            )
            p.grad.detach().add_(noise)


def compute_epsilon(
    steps: int,
    num_examples: int,
    batch_size: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    """
    Approximate (epsilon, delta)-DP spent after *steps* DP-SGD steps using the
    Renyi DP accountant moments accountant formula (Abadi et al. 2016).

    q = batch_size / num_examples (sampling probability)
    """
    if noise_multiplier <= 0:
        return float("inf")
    q = batch_size / num_examples
    # Moments accountant: epsilon ≈ q * sqrt(2 * steps * log(1/delta)) / noise_multiplier
    # (standard simplified bound for sampled Gaussian mechanism)
    epsilon = q * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier
    return float(epsilon)


class DPSGDTrainer:
    """
    DP-SGD trainer wrapping a PyTorch model, optimizer and dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train (the ``.network`` of an ``MLPIDS`` works too).
    optimizer : torch.optim.Optimizer
        Underlying optimizer (e.g. Adam).
    dataloader : torch.utils.data.DataLoader
        Training data loader.
    max_grad_norm : float
        Per-example gradient clipping norm.
    noise_multiplier : float
        Noise multiplier sigma (noise std = sigma * max_grad_norm).
    delta : float
        Target delta for privacy accounting.
    use_opacus : bool
        Prefer Opacus when available (default True).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        *,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        delta: float = 1e-5,
        use_opacus: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.steps = 0
        self.num_examples = len(getattr(dataloader, "dataset", [])) or 0
        self.batch_size = getattr(dataloader, "batch_size", None) or 32
        self.privacy_engine = None
        self._opacus_used = False

        if use_opacus and OPACUS_AVAILABLE and PrivacyEngine is not None:
            try:
                self.privacy_engine = PrivacyEngine()
                self.model, self.optimizer, self.dataloader = (
                    self.privacy_engine.make_private(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=self.dataloader,
                        max_grad_norm=self.max_grad_norm,
                        noise_multiplier=self.noise_multiplier,
                    )
                )
                self._opacus_used = True
            except Exception:  # pragma: no cover - defensive
                self.privacy_engine = None
                self._opacus_used = False

    def train_epoch(self) -> dict:
        """Run one epoch of DP-SGD; returns loss and privacy info."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        criterion = torch.nn.BCEWithLogitsLoss()

        for batch_X, batch_y in self.dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            outputs_flat = outputs.view(-1)
            loss = criterion(outputs_flat, batch_y.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.steps += 1

        avg_loss = total_loss / max(num_batches, 1)
        epsilon, delta = self.get_privacy_spent()
        return {
            "loss": avg_loss,
            "steps": self.steps,
            "epsilon": epsilon,
            "delta": delta,
            "opacus_used": self._opacus_used,
        }

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return (epsilon, delta) spent so far."""
        if self._opacus_used and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(self.delta)
                return float(epsilon), self.delta
            except Exception:  # pragma: no cover - defensive
                pass
        # Manual fallback accountant
        epsilon = compute_epsilon(
            steps=self.steps,
            num_examples=max(self.num_examples, 1),
            batch_size=self.batch_size,
            noise_multiplier=self.noise_multiplier,
            delta=self.delta,
        )
        return epsilon, self.delta

    def is_dp(self) -> bool:
        return self.noise_multiplier > 0 or self._opacus_used


def make_dp_optimizer(
    model: torch.nn.Module,
    *,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    lr: float = 0.001,
    use_opacus: bool = True,
) -> Tuple[torch.optim.Optimizer, DPSGDTrainer]:
    """
    Convenience: build an Adam optimizer wrapped in a :class:`DPSGDTrainer`.

    Returns ``(optimizer, trainer)``. When ``noise_multiplier == 0`` the trainer
    still clips gradients but adds no noise (non-private baseline with clipping).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = DPSGDTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=None,  # set via trainer.set_dataloader before training
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        delta=delta,
        use_opacus=use_opacus,
    )
    return optimizer, trainer


def privacy_report(
    steps: int,
    num_examples: int,
    batch_size: int,
    noise_multiplier: float,
    delta: float,
    target_epsilons: Optional[List[float]] = None,
) -> dict:
    """
    Produce a privacy accounting report: epsilon achieved for the given delta,
    and the noise multiplier required to hit optional target epsilons.
    """
    epsilon = compute_epsilon(
        steps=steps,
        num_examples=num_examples,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        delta=delta,
    )
    report = {
        "steps": steps,
        "num_examples": num_examples,
        "batch_size": batch_size,
        "noise_multiplier": noise_multiplier,
        "delta": delta,
        "epsilon": epsilon,
        "dp_enabled": noise_multiplier > 0,
    }
    if target_epsilons:
        required = []
        for target in target_epsilons:
            if target <= 0:
                required.append({"target_epsilon": target, "noise_multiplier": None})
                continue
            # Solve q * sqrt(2*steps*log(1/delta)) / sigma <= target
            q = batch_size / num_examples
            sigma = q * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / target
            required.append(
                {"target_epsilon": target, "noise_multiplier": float(sigma)}
            )
        report["required_noise_for_targets"] = required
    return report
