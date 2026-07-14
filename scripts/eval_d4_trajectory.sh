#!/usr/bin/env bash
# D4 Checkpoint Trajectory Evaluation
# Evaluates all saved checkpoints for trace validity trajectory

set -euo pipefail

EXPERIMENT_ROOT="${1:-runs/d4_specialist_ablation}"
OUTPUT_FILE="${2:-evaluations/d4_checkpoint_trajectory.json}"

echo "Evaluating checkpoints in: $EXPERIMENT_ROOT"

.venv/bin/python << 'PYEOF'
import json, os, subprocess, sys
from pathlib import Path

exp_root = Path(os.environ.get("EXP_ROOT", "runs/d4_specialist_ablation"))
output = Path(os.environ.get("OUT_FILE", "evaluations/d4_checkpoint_trajectory.json"))

experiments = ["sum_a", "sum_b", "sum_c", "neg_a", "neg_b", "neg_c"]
operators = {
    "sum_a": "aggregation.sum", "sum_b": "aggregation.sum", "sum_c": "aggregation.sum",
    "neg_a": "scalar.neg", "neg_b": "scalar.neg", "neg_c": "scalar.neg",
}

results = {}

for exp in experiments:
    ckpt_dir = exp_root / exp / "seed_0" / operators[exp].replace(".", "_") / "checkpoints"
    if not ckpt_dir.exists():
        print(f"Missing: {ckpt_dir}")
        continue
    
    results[exp] = {}
    
    for ckpt in sorted(ckpt_dir.glob("*.pt")):
        step = ckpt.stem.replace("step_", "").replace("final", "3000").lstrip("0") or "0"
        step = int(step)
        
        # Create manifest pointing to this checkpoint
        manifest = {
            "unit_checkpoints": {operators[exp]: str(ckpt)},
            "experiment_fingerprint": "d4_specialist_ablation"
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest, f)
            manifest_path = f.name
        
        try:
            out = subprocess.run([
                ".venv/bin/opfusion-diagnose-specialist-failures",
                "--config", f"configs/experiments/d4_specialist_ablation/{exp}.yaml",
                "--manifest", manifest_path,
                "--operators", operators[exp],
                "--split", "validation",
                "--evaluation-seed", "704000",
                "--examples-per-operator", "64",
                "--retain-examples", "0",
                "--out", "/dev/stdout"
            ], capture_output=True, text=True, timeout=120, cwd=".")
            
            if out.returncode == 0:
                data = json.loads(out.stdout)
                op_data = data["operators"][operators[exp]]
                results[exp][step] = {
                    "trace_validity": op_data["generation"]["trace_validity"],
                    "final_value_accuracy": op_data["generation"]["final_value_accuracy"],
                    "stop_accuracy": op_data["generation"]["stop_accuracy"],
                    "tf_token_accuracy": op_data["teacher_forced"]["token_accuracy"],
                    "tf_first_token_accuracy": op_data["teacher_forced"]["first_token_accuracy"],
                    "tf_seq_accuracy": op_data["teacher_forced"]["sequence_argmax_accuracy"],
                }
                print(f"  {exp} step {step}: trace={results[exp][step]['trace_validity']:.3f}")
            else:
                print(f"  {exp} step {step}: FAILED - {out.stderr[:200]}")
        finally:
            os.unlink(manifest_path)

output.parent.mkdir(parents=True, exist_ok=True)
with open(output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved trajectory to {output}")
PYEOF