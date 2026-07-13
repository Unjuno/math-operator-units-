from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import GPTConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    loss: float | None = None,
    config: GPTConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if loss is not None:
        state["loss"] = loss
    if config is not None:
        state["config"] = config.to_dict()
    if extra:
        state.update(extra)
    torch.save(state, p)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    p = Path(path)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(p, map_location=dev, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


class CheckpointManager:
    def __init__(self, output_dir: str | Path, keep_last_k: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self._saved_paths: list[Path] = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        loss: float | None = None,
        config: GPTConfig | None = None,
        extra: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> Path:
        name = f"{label or f'step_{step}'}.pt"
        path = self.output_dir / name
        save_checkpoint(path, model, optimizer, step, loss, config, extra)
        self._saved_paths.append(path)
        if len(self._saved_paths) > self.keep_last_k:
            old = self._saved_paths.pop(0)
            if old.exists():
                old.unlink()
        return path

    def save_best(self, model: torch.nn.Module, **kwargs: Any) -> Path:
        return self.save(model, label="best", **kwargs)

    def save_final(self, model: torch.nn.Module, **kwargs: Any) -> Path:
        return self.save(model, label="final", **kwargs)
