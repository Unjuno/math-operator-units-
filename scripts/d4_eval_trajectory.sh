#!/bin/bash
# D4 Checkpoint Trajectory Evaluation
# Evaluates all checkpoints for a specialist across the trajectory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

source .venv/bin/activate

usage() {
    echo "Usage: $0 <operator> <output_dir>"
    echo "  operator: aggregation.sum or scalar.neg"
    echo "  output_dir: runs/d4_specialist_ablation/..."
    exit 1
}

OPERATOR="${1:-}"
OUTPUT_DIR="${2:-}"

[[ -z "$OPERATOR" ]] && usage
[[ -z "$OUTPUT_DIR" ]] && usage

if [[ ! -d "$OUTPUT_DIR/seed_0/${OPERATOR//./_}/checkpoints" ]]; then
    echo "Checkpoint directory not found: $OUTPUT_DIR/seed_0/${OPERATOR//./_}/checkpoints"
    exit 1
fi

# Fixed config for evaluation (use the pilot config for manifest generation)
CONFIG="configs/experiments/model_design_pilot_weak_retention.yaml"
MANIFEST="runs/d4_specialist_ablation/${OPERATOR//./_}/seed_0/fusion_subsets/subset_31.json"

# If no manifest, create one from the specialist checkpoint
if [[ ! -f "$MANIFEST" ]]; then
    echo "Creating manifest..."
    mkdir -p "$(dirname "$MANIFEST")"
    cat > "$MANIFEST" <<EOF
{
  "experiment_fingerprint": "d4_specialist_ablation_${OPERATOR//./_}",
  "unit_checkpoints": {
    "${OPERATOR}": ""
  },
  "subset_id": "subset_31"
}
EOF
fi

# Find all checkpoints
CHECKPOINT_DIR="$OUTPUT_DIR/seed_0/${OPERATOR//./_}/checkpoints"
CKPTS=($(find "$CHECKPOINT_DIR" -name "*.pt" -type f | sort -V))

echo "=== Evaluating ${#CKPTS[@]} checkpoints for $OPERATOR ==="
echo ""

for ckpt in "${CKPTS[@]}"; do
    step=$(basename "$ckpt" .pt | sed 's/step_//' | sed 's/^0*//')
    [[ -z "$step" ]] && step=0
    [[ "$step" == "final" ]] && step=3000
    
    echo "Step $step: $ckpt"
    
    # Update manifest with this checkpoint
    TMP_MANIFEST=$(mktemp)
    cat > "$TMP_MANIFEST" <<EOF
{
  "experiment_fingerprint": "d4_specialist_ablation_${OPERATOR//./_}",
  "unit_checkpoints": {
    "${OPERATOR}": "$ckpt"
  },
  "subset_id": "subset_31"
}
EOF
    
    OUT_FILE="evaluations/d4_specialist_ablation/${OPERATOR//./_}_step_${step}.json"
    mkdir -p "$(dirname "$OUT_FILE")"
    
    .venv/bin/opfusion-diagnose-specialist-failures \
        --config "$CONFIG" \
        --manifest "$TMP_MANIFEST" \
        --operators "$OPERATOR" \
        --split validation \
        --evaluation-seed 704000 \
        --examples-per-operator 64 \
        --retain-examples 0 \
        --out "$OUT_FILE" \
        2>&1 | tail -20
    
    rm -f "$TMP_MANIFEST"
done

echo ""
echo "=== Trajectory evaluation complete ==="
echo "Results in: evaluations/d4_specialist_ablation/"