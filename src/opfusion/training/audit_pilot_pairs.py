from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .data import EXPERIMENT_OPERATORS


PAIRS = (
    ("identity_unanchored", "identity_retention"),
    ("weak_unanchored", "weak_retention"),
)
SHARED_JOBS = ("base.common", "joint.all_five.exposure_matched")


def _job_dir(root: Path, condition: str, job_id: str) -> Path:
    return root / condition / "seed_0" / job_id.replace(".", "_")


def _resolve(repo_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _selected_checkpoint(repo_root: Path, root: Path, condition: str, job_id: str) -> tuple[Path, dict[str, Any]]:
    complete_path = _job_dir(root, condition, job_id) / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_checkpoint")
    if selected is None:
        raise RuntimeError(f"selected checkpoint missing: {complete_path}")
    path = _resolve(repo_root, selected)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, payload


def _state_hash(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"model_state_dict missing: {checkpoint}")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _runtime_microbatch(root: Path, condition: str, job_id: str) -> int | None:
    path = _job_dir(root, condition, job_id) / "runtime_state.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("micro_batch_size")
    return int(value) if value is not None else None


def audit_pilot_pairs(repo_root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    pilot_root = repo_root / "runs/model_design_pilot"
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []

    for left, right in PAIRS:
        shared: dict[str, Any] = {}
        for job_id in SHARED_JOBS:
            try:
                left_path, left_complete = _selected_checkpoint(repo_root, pilot_root, left, job_id)
                right_path, right_complete = _selected_checkpoint(repo_root, pilot_root, right, job_id)
                left_hash = _state_hash(left_path)
                right_hash = _state_hash(right_path)
                identical = left_hash == right_hash
                shared[job_id] = {
                    "identical_model_state": identical,
                    "left_checkpoint": str(left_path),
                    "right_checkpoint": str(right_path),
                    "left_step": left_complete.get("selected_step"),
                    "right_step": right_complete.get("selected_step"),
                    "left_sha256": left_hash,
                    "right_sha256": right_hash,
                }
                if not identical:
                    errors.append(
                        {
                            "kind": "paired_shared_endpoint_mismatch",
                            "pair": [left, right],
                            "job_id": job_id,
                            "left_sha256": left_hash,
                            "right_sha256": right_hash,
                        }
                    )
            except Exception as exc:
                shared[job_id] = {"error": repr(exc)}
                errors.append(
                    {
                        "kind": "paired_shared_endpoint_unreadable",
                        "pair": [left, right],
                        "job_id": job_id,
                        "error": repr(exc),
                    }
                )

        microbatch: dict[str, Any] = {}
        for operator in EXPERIMENT_OPERATORS:
            left_value = _runtime_microbatch(pilot_root, left, operator)
            right_value = _runtime_microbatch(pilot_root, right, operator)
            microbatch[operator] = {"left": left_value, "right": right_value, "equal": left_value == right_value}
            if left_value is not None and right_value is not None and left_value != right_value:
                warnings.append(
                    {
                        "kind": "paired_specialist_microbatch_mismatch",
                        "pair": [left, right],
                        "operator": operator,
                        "left": left_value,
                        "right": right_value,
                        "interpretation": (
                            "effective batch is still matched, but gradient accumulation and floating-point order differ"
                        ),
                    }
                )

        pair_results.append(
            {
                "left": left,
                "right": right,
                "shared_endpoint_checks": shared,
                "specialist_microbatch_checks": microbatch,
            }
        )

    return {
        "status": "passed" if not errors else "failed",
        "pilot_root": str(pilot_root),
        "pairs": pair_results,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that retention/unanchored pilot pairs share exact base and joint endpoints"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="audits/model_design_pilot/pair_consistency.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = audit_pilot_pairs(args.repo_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)
    else:
        print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
