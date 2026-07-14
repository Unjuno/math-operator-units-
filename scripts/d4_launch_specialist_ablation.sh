#!/bin/bash
# D4 Specialist Ablation Launcher (parallel, after base is trained)
# Runs 6 specialist-only experiments for aggregation.sum and scalar.neg
#
# Prerequisite: base.common must be trained first via run_d4_ablation.sh --base-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="runs/d4_specialist_ablation"
BASE_OUTPUT_DIR="$BASE_DIR/base"

source .venv/bin/activate

# Check base is trained
if [[ ! -f "$BASE_OUTPUT_DIR/seed_0/base_common/complete.json" ]]; then
    echo "ERROR: Base checkpoint not found at $BASE_OUTPUT_DIR"
    echo "Run './scripts/run_d4_ablation.sh --base-only' first."
    exit 1
fi

job_for_config() {
    local name
    name="$(basename "$1" .yaml)"
    case "$name" in
        sum_*) echo "aggregation.sum" ;;
        neg_*) echo "scalar.neg" ;;
        *)     echo "ERROR: unknown config $1" >&2; exit 1 ;;
    esac
}

CONFIGS=(
    "configs/experiments/d4_specialist_ablation/sum_a.yaml"
    "configs/experiments/d4_specialist_ablation/sum_b.yaml"
    "configs/experiments/d4_specialist_ablation/sum_c.yaml"
    "configs/experiments/d4_specialist_ablation/neg_a.yaml"
    "configs/experiments/d4_specialist_ablation/neg_b.yaml"
    "configs/experiments/d4_specialist_ablation/neg_c.yaml"
)

echo "=== D4 Specialist Ablation Launcher (parallel) ==="
echo "Starting ${#CONFIGS[@]} experiments..."
echo ""

for config in "${CONFIGS[@]}"; do
    JOB="$(job_for_config "$config")"
    NAME="$(basename "$config" .yaml)"
    OUTPUT_DIR="$BASE_DIR/$NAME"

    # Stage base checkpoint
    mkdir -p "$OUTPUT_DIR/seed_0"
    if [[ -d "$BASE_OUTPUT_DIR/seed_0/base_common" ]] && [[ ! -d "$OUTPUT_DIR/seed_0/base_common" ]]; then
        cp -a "$BASE_OUTPUT_DIR/seed_0/base_common" "$OUTPUT_DIR/seed_0/base_common"
    fi

    echo ">>> Launching: $NAME ($JOB)"
    .venv/bin/opfusion-train-one-design \
        --config "$config" \
        --job "$JOB" \
        --seed 0 &
    sleep 2  # Stagger starts
done

wait
echo ""
echo "=== All D4 experiments completed ==="