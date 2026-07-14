#!/usr/bin/env .venv/bin/python
"""D4 Checkpoint Trajectory Evaluation"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from opfusion.specialist_failure_diagnostics import diagnose_manifest


def eval_trajectory(operator: str, output_dir: Path, config_path: Path, out_path: Path):
    ckpt_dir = output_dir / "seed_0" / operator.replace(".", "_") / "checkpoints"
    
    if not ckpt_dir.exists():
        print(f"Checkpoint dir not found: {ckpt_dir}")
        return
    
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    print(f"Found {len(ckpts)} checkpoints for {operator}")
    
    results = {}
    
    for ckpt in ckpts:
        step_name = ckpt.stem
        step = step_name.replace("step_", "").replace("final", "3000")
        try:
            step = int(step)
        except ValueError:
            step = 0
        
        print(f"  Evaluating {operator} step {step}...")
        
        # Create temporary manifest
        manifest = {
            "experiment_fingerprint": f"d4_specialist_ablation_{operator.replace('.', '_')}",
            "unit_checkpoints": {operator: str(ckpt)},
            "subset_id": "subset_31",
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest, f)
            manifest_path = f.name
        
        try:
            # Use diagnose_manifest directly
            report = diagnose_manifest(
                config_path=str(config_path),
                manifest_path=manifest_path,
                operators=[operator],
                split="validation",
                evaluation_seed=704000,
                examples_per_operator=64,
                device_name="cuda",
            )
            
            op_data = report["operators"][operator]
            results[step] = {
                "trace_validity": op_data["generation"]["trace_validity"],
                "final_value_accuracy": op_data["generation"]["final_value_accuracy"],
                "stop_accuracy": op_data["generation"]["stop_accuracy"],
                "tf_token_accuracy": op_data["teacher_forced"]["token_accuracy"],
                "tf_first_token_accuracy": op_data["teacher_forced"]["first_token_accuracy"],
                "tf_seq_accuracy": op_data["teacher_forced"]["sequence_argmax_accuracy"],
                "checkpoint": str(ckpt),
            }
            print(f"    trace_validity={results[step]['trace_validity']:.3f} "
                  f"final_acc={results[step]['final_value_accuracy']:.3f} "
                  f"tf_tok={results[step]['tf_token_accuracy']:.3f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[step] = {"error": str(e)}
        finally:
            os.unlink(manifest_path)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {operator} trajectory to {out_path}")


def main():
    base = Path(__file__).parents[1]
    
    config = base / "configs/experiments/model_design_pilot_weak_retention.yaml"
    
    # SUM trajectory
    eval_trajectory(
        "aggregation.sum",
        base / "runs/d4_specialist_ablation/sum_c",
        config,
        base / "evaluations/d4_specialist_ablation/sum_c_trajectory.json"
    )
    
    # NEG trajectory
    eval_trajectory(
        "scalar.neg",
        base / "runs/d4_specialist_ablation/neg_c",
        config,
        base / "evaluations/d4_specialist_ablation/neg_c_trajectory.json"
    )


if __name__ == "__main__":
    main()