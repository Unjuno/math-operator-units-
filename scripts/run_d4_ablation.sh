#!/usr/bin/env bash
# D4 Specialist-Only Ablation Launcher
# Runs 6 experiments: SUM-A, SUM-B, SUM-C, NEG-A, NEG-B, NEG-C
#
# Usage:
#   ./scripts/run_d4_ablation.sh              # full run (base + 6 conditions)
#   ./scripts/run_d4_ablation.sh --skip-base   # skip base pre-training
#   ./scripts/run_d4_ablation.sh --base-only   # train base only, then exit

set -euo pipefail

BASE_DIR="runs/d4_specialist_ablation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_BASE=false
BASE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --skip-base) SKIP_BASE=true ;;
        --base-only) BASE_ONLY=true ;;
    esac
done

CONFIGS=(
    "configs/experiments/d4_specialist_ablation/sum_a.yaml"
    "configs/experiments/d4_specialist_ablation/sum_b.yaml"
    "configs/experiments/d4_specialist_ablation/sum_c.yaml"
    "configs/experiments/d4_specialist_ablation/neg_a.yaml"
    "configs/experiments/d4_specialist_ablation/neg_b.yaml"
    "configs/experiments/d4_specialist_ablation/neg_c.yaml"
)

# Map config basename → operator job ID
job_for_config() {
    local name
    name="$(basename "$1" .yaml)"
    case "$name" in
        sum_*) echo "aggregation.sum" ;;
        neg_*) echo "scalar.neg" ;;
        *)     echo "ERROR: unknown config $1" >&2; exit 1 ;;
    esac
}

echo "=========================================="
echo "D4 Specialist-Only Ablation Launcher"
echo "=========================================="
echo "Will run ${#CONFIGS[@]} experiments:"
for cfg in "${CONFIGS[@]}"; do
    echo "  $cfg → $(job_for_config "$cfg")"
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

cd "$REPO_ROOT"
mkdir -p logs "$BASE_DIR"

_strip_base_fingerprints() {
    local dir="$1"
    local selected_pt="$dir/seed_0/base_common/selected.pt"
    local complete_json="$dir/seed_0/base_common/complete.json"
    if [[ -f "$selected_pt" ]]; then
        echo "  Stripping fingerprint from base selected.pt..."
        .venv/bin/python -c "
import torch
pt = '$selected_pt'
payload = torch.load(pt, map_location='cpu', weights_only=False)
payload.pop('experiment_fingerprint', None)
torch.save(payload, pt)
"
    fi
    if [[ -f "$complete_json" ]]; then
        echo "  Stripping fingerprint from base complete.json..."
        .venv/bin/python -c "
import json
path = '$complete_json'
data = json.loads(open(path).read())
data.pop('experiment_fingerprint', None)
open(path, 'w').write(json.dumps(data, indent=2) + '\n')
"
    fi
}

# ── Step 0: Train base.common once into a shared location ──────────────────
BASE_OUTPUT_DIR="$BASE_DIR/base"
if [[ "$SKIP_BASE" == true ]]; then
    echo ""
    echo "--- Skipping base pre-training (--skip-base) ---"
elif [[ -f "$BASE_OUTPUT_DIR/seed_0/base_common/complete.json" ]]; then
    echo ""
    echo "--- Base already trained at $BASE_OUTPUT_DIR ---"
else
    echo ""
    echo "=========================================="
    echo "Step 0: Training shared base.common"
    echo "=========================================="
    FIRST_CFG="${CONFIGS[0]}"
    .venv/bin/opfusion-train-one-design \
        --config "$FIRST_CFG" \
        --job base.common \
        --seed 0 \
        2>&1 | tee "logs/d4_base_$(date +%Y%m%d_%H%M%S).log"
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "✗ Base training FAILED"
        exit 1
    fi
    echo "✓ Base training complete"

    # Move base to shared location and strip fingerprints
    mkdir -p "$BASE_OUTPUT_DIR/seed_0"
    FIRST_OUTPUT_DIR="$BASE_DIR/sum_a"
    if [[ -d "$FIRST_OUTPUT_DIR/seed_0/base_common" ]]; then
        cp -a "$FIRST_OUTPUT_DIR/seed_0/base_common" "$BASE_OUTPUT_DIR/seed_0/"
    fi
    _strip_base_fingerprints "$BASE_OUTPUT_DIR"
fi

if [[ "$BASE_ONLY" == true ]]; then
    echo ""
    echo "--- Base-only mode: exiting ---"
    exit 0
fi

# ── Step 1-6: Run each condition ──────────────────────────────────────────
for cfg in "${CONFIGS[@]}"; do
    JOB="$(job_for_config "$cfg")"
    NAME="$(basename "$cfg" .yaml)"
    OUTPUT_DIR="$BASE_DIR/$NAME"
    
    echo ""
    echo "=========================================="
    echo "Starting: $NAME ($JOB)"
    echo "Config: $cfg"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="

    # Stage base checkpoint into this condition's output dir
    mkdir -p "$OUTPUT_DIR/seed_0"
    if [[ -d "$BASE_OUTPUT_DIR/seed_0/base_common" ]] && [[ ! -d "$OUTPUT_DIR/seed_0/base_common" ]]; then
        echo "Staging shared base checkpoint..."
        cp -a "$BASE_OUTPUT_DIR/seed_0/base_common" "$OUTPUT_DIR/seed_0/base_common"
        _strip_base_fingerprints "$OUTPUT_DIR"
    fi
    
    .venv/bin/opfusion-train-one-design \
        --config "$cfg" \
        --job "$JOB" \
        --seed 0 \
        2>&1 | tee "logs/d4_${NAME}_$(date +%Y%m%d_%H%M%S).log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "✓ Completed: $NAME"
    else
        echo "✗ FAILED: $NAME"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All D4 experiments completed successfully"
echo "=========================================="