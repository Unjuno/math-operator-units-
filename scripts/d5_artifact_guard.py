from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from opfusion.training.design_config import load_design_run_config
from opfusion.training.experiment_contract import build_contract

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/d5_operator_unit_factory"


@dataclass(frozen=True)
class Target:
    config: str
    output: Path
    job_dir: Path


TARGETS = (
    Target(
        "configs/experiments/d5_operator_unit_factory/shared_base.yaml",
        RUN_ROOT / "shared_base",
        RUN_ROOT / "shared_base/seed_0/base_common",
    ),
    Target(
        "configs/experiments/d5_operator_unit_factory/sum_scratch.yaml",
        RUN_ROOT / "sum_scratch",
        RUN_ROOT / "sum_scratch/seed_0/aggregation_sum",
    ),
    Target(
        "configs/experiments/d5_operator_unit_factory/sum_base.yaml",
        RUN_ROOT / "sum_base",
        RUN_ROOT / "sum_base/seed_0/aggregation_sum",
    ),
    Target(
        "configs/experiments/d5_operator_unit_factory/neg_scratch.yaml",
        RUN_ROOT / "neg_scratch",
        RUN_ROOT / "neg_scratch/seed_0/scalar_neg",
    ),
    Target(
        "configs/experiments/d5_operator_unit_factory/neg_base.yaml",
        RUN_ROOT / "neg_base",
        RUN_ROOT / "neg_base/seed_0/scalar_neg",
    ),
)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def expected_fingerprint(config_path: str) -> str:
    config = load_design_run_config(ROOT / config_path)
    config.validate()
    return str(build_contract(ROOT, config)["fingerprint"])


def verify(target: Target) -> None:
    complete_path = target.job_dir / "complete.json"
    selected_path = target.job_dir / "selected.pt"
    if not complete_path.exists() and not selected_path.exists():
        return
    if not complete_path.is_file() or not selected_path.is_file():
        raise RuntimeError(f"partial D5 output requires inspection: {target.job_dir}")
    contract_path = target.output / "experiment_contract.json"
    if not contract_path.is_file():
        raise RuntimeError(f"missing experiment contract: {contract_path}")
    expected = expected_fingerprint(target.config)
    contract = load_json(contract_path)
    complete = load_json(complete_path)
    checkpoint = torch.load(selected_path, map_location="cpu", weights_only=False)
    observed = {
        "contract": contract.get("fingerprint"),
        "complete": complete.get("experiment_fingerprint"),
        "checkpoint": checkpoint.get("experiment_fingerprint"),
    }
    if any(value != expected for value in observed.values()):
        raise RuntimeError(
            f"stale or mixed D5 output: {target.job_dir}; expected={expected}; observed={observed}"
        )


def main() -> int:
    for target in TARGETS:
        verify(target)
    print("D5 artifact guard: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
