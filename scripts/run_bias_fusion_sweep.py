#!/usr/bin/env python3
"""Bias Fusion Sweep: extract logit biases from Model Factory checkpoints
and evaluate fusion strategies across model sizes and data amounts.

Usage:
    python scripts/run_bias_fusion_sweep.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from opfusion.model import GPTConfig, GPTModel
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory
from opfusion.training.design_config import load_design_run_config

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_LABELS = ["4K", "16K", "65K", "262K", "1M"]
SNAPSHOT_STEPS = [39, 157, 637, 2569, 9804]
SIZES = ["nano", "small", "medium", "1m"]
SIZE_LABELS = {"nano": "125K", "small": "250K", "medium": "500K", "1m": "1M"}
OPERATORS = list(EXPERIMENT_OPERATORS)

EVALUATION_SEED = 702_000
EXAMPLES_PER_OPERATOR = 64
MAX_NEW_TOKENS = 256


def _load_model(path: Path, device: torch.device) -> GPTModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(payload["model_config"])
    model = GPTModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def extract_biases(
    model: GPTModel,
    factory: SyntheticTraceFactory,
    device: torch.device,
    *,
    seed: int = EVALUATION_SEED,
    examples: int = EXAMPLES_PER_OPERATOR,
) -> dict[str, dict]:
    """Extract teacher-forced logits per operator and compute bias vectors."""
    tokenizer = factory.tokenizer
    results: dict[str, dict] = {}

    with torch.no_grad():
        for op_idx, operator_id in enumerate(OPERATORS):
            all_logits: list[torch.Tensor] = []
            for sample_idx in range(examples):
                example = factory.training_example(
                    operator_id,
                    seed=seed,
                    split="validation",
                    step=op_idx,
                    sample_index=sample_idx,
                )
                prompt = tokenizer.encode_tokens(example.prompt_tokens, add_bos=True, add_eos=False)
                expected = tokenizer.encode_tokens(example.response_tokens, add_bos=False, add_eos=True)
                sequence = prompt + expected
                input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
                response_start = len(prompt) - 1

                logits = model(input_ids)[:, response_start:, :]
                all_logits.append(logits.squeeze(0).cpu())

            all_logits = torch.stack(all_logits)
            results[operator_id] = {
                "mean_logits": all_logits.mean(dim=0),
                "std_logits": all_logits.std(dim=0),
                "n_examples": examples,
            }

    return results


def compute_bias(operator_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    return operator_logits - base_logits


def bias_norm(bias: torch.Tensor) -> float:
    return float(bias.norm().item())


def operator_similarity(bias_a: torch.Tensor, bias_b: torch.Tensor) -> float:
    a = bias_a.flatten()
    b = bias_b.flatten()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def fusion_residual(
    fused_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> float:
    return float(F.mse_loss(fused_logits, target_logits).item())


def teacher_accuracy(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == tokens).sum().item()
    total = tokens.numel()
    return correct / max(1, total)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bias Fusion Sweep")
    parser.add_argument("--dry-run", action="store_true", help="print plan and exit")
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--examples", type=int, default=EXAMPLES_PER_OPERATOR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    summary_rows: list[dict] = []

    for size in SIZES:
        config_path = ROOT / f"configs/experiments/bias_factory/{size}_config.yaml"
        if not config_path.exists():
            print(f"SKIP: no config for {size} at {config_path}", file=sys.stderr)
            continue

        run = load_design_run_config(config_path)
        tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
        factory = SyntheticTraceFactory(tokenizer, run.data)

        for snap_label, snap_step in zip(SNAPSHOT_LABELS, SNAPSHOT_STEPS):
            checkpoint_path = (
                ROOT / f"runs/bias_factory/{size}/seed_0/base_common/checkpoints/step_{snap_step:09d}.pt"
            )
            if not checkpoint_path.exists():
                print(f"SKIP: {size}/{snap_label} (step {snap_step}) not found", file=sys.stderr)
                continue

            if args.dry_run:
                print(f"  WOULD PROCESS: {size}/{snap_label} ({checkpoint_path})")
                continue

            print(f"Processing: {size}/{snap_label}...", file=sys.stderr)
            model = _load_model(checkpoint_path, device)

            biases = extract_biases(model, factory, device, seed=args.seed, examples=args.examples)

            for operator_id in OPERATORS:
                op_bias = biases[operator_id]["mean_logits"]
                row = {
                    "model_size": SIZE_LABELS[size],
                    "training_examples": snap_label,
                    "operator": operator_id,
                    "bias_norm": bias_norm(op_bias),
                    "bias_std": float(biases[operator_id]["std_logits"].mean().item()),
                }
                summary_rows.append(row)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.dry_run and summary_rows:
        out_path = ROOT / "evaluations/bias_fusion_sweep/summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "description": "Bias Fusion Sweep summary",
            "seed": args.seed,
            "examples_per_operator": args.examples,
            "rows": summary_rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
