from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opfusion.io import load_yaml

_LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"]
_LEVEL_RANK = {level: rank for rank, level in enumerate(_LEVEL_ORDER)}


def level_rank(level: str) -> int:
    try:
        return _LEVEL_RANK[level]
    except KeyError as exc:
        raise ValueError(f"unknown applicability level: {level!r}") from exc


@dataclass(frozen=True)
class ApplicabilityPolicy:
    raw: dict[str, Any]

    @property
    def core_rule(self) -> str:
        return str(self.raw["core_rule"])

    def minimum_level_for_mode(self, implementation_mode: str) -> str:
        try:
            return self.raw["implementation_mode_rules"][implementation_mode]["minimum_level"]
        except KeyError as exc:
            raise ValueError(f"unknown implementation mode: {implementation_mode!r}") from exc

    def can_enter_mode(self, level: str, implementation_mode: str) -> bool:
        return level_rank(level) >= level_rank(self.minimum_level_for_mode(implementation_mode))

    def can_enter_runtime_fusion(self, level: str) -> bool:
        required = self.raw["runtime_fusion_rule"]["minimum_level"]
        return level_rank(level) >= level_rank(required)

    def supervision_is_trainable(self, supervision_source: str) -> bool | str:
        try:
            return self.raw["supervision_sources"][supervision_source]["trainable"]
        except KeyError as exc:
            raise ValueError(f"unknown supervision source: {supervision_source!r}") from exc


def load_applicability_policy(path: str | Path) -> ApplicabilityPolicy:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("applicability policy must be a mapping")
    for required in ("core_rule", "levels", "supervision_sources", "implementation_mode_rules", "runtime_fusion_rule"):
        if required not in data:
            raise ValueError(f"applicability policy missing required section: {required}")
    return ApplicabilityPolicy(raw=data)
