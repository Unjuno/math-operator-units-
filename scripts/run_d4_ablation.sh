#!/usr/bin/env bash
# D4 Specialist-Only Ablation Launcher
# Runs 6 experiments: SUM-A, SUM-B, SUM-C, NEG-A, NEG-B, NEG-C

set -euo pipefail

CONFIGS=(
    "configs/experiments/d4_specialist_ablation/sum_a.yaml"
    "configs/experiments/d4_specialist_ablation/sum_b.yaml"
    "configs/experiments/d4_specialist_ablation/sum_c.yaml"
    "configs/experiments/d4_specialist_ablation/neg_a.yaml"
    "configs/experiments/d4_specialist_ablation/neg_b.yaml"
    "configs/experiments/d4_specialist_ablation/neg_c.yaml"
)

echo "=========================================="
echo "D4 Specialist-Only Ablation Launcher"
echo "=========================================="
echo "Will run ${#CONFIGS[@]} experiments:"
for cfg in "${CONFIGS[@]}"; do
    echo "  $cfg"
done
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found, CUDA may not be available"
fi

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "Activated .venv"
fi

# Run each experiment
for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting: $cfg"
    echo "=========================================="
    
    .venv/bin/opfusion-train-one-design \
        --config "$cfg" \
        --seed 0 \
        --output-root runs/d4_specialist_ablation \
        2>&1 | tee "logs/d4_$(basename "$cfg" .yaml)_$(date +%Y%m%d_%H%M%S).log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "✓ Completed: $cfg"
    else
        echo "✗ FAILED: $cfg"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All D4 experiments completed successfully"
echo "=========================================="