#!/bin/bash
# D4 Specialist Ablation Launcher
# Runs 6 specialist-only experiments for aggregation.sum and scalar.neg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# Activate venv
source .venv/bin/activate

# Experiment configs
CONFIGS=(
    "configs/experiments/d4_specialist_ablation/sum_a.yaml"
    "configs/experiments/d4_specialist_ablation/sum_b.yaml"
    "configs/experiments/d4_specialist_ablation/sum_c.yaml"
    "configs/experiments/d4_specialist_ablation/neg_a.yaml"
    "configs/experiments/d4_specialist_ablation/neg_b.yaml"
    "configs/experiments/d4_specialist_ablation/neg_c.yaml"
)

echo "=== D4 Specialist Ablation Launcher ==="
echo "Starting 6 experiments..."
echo ""

for config in "${CONFIGS[@]}"; do
    echo ">>> Launching: $config"
    .venv/bin/opfusion-train-one-design --config "$config" &
    sleep 2  # Stagger starts
done

wait
echo ""
echo "=== All D4 experiments completed ==="