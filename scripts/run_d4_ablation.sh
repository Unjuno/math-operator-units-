#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="runs/d4_specialist_ablation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SKIP_BASE=false
BASE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --skip-base) SKIP_BASE=true ;;
        --base-only) BASE_ONLY=true ;;
        *) echo "unknown argument: $arg" >&2; exit 64 ;;
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
BASE_CONFIG="${CONFIGS[0]}"
BASE_OUTPUT_DIR="$BASE_DIR/base"
BASE_SOURCE="$BASE_OUTPUT_DIR/seed_0/base_common"

job_for_config() {
    case "$(basename "$1" .yaml)" in
        sum_*) echo "aggregation.sum" ;;
        neg_*) echo "scalar.neg" ;;
        *) echo "unknown D4 config: $1" >&2; exit 64 ;;
    esac
}

for executable in .venv/bin/python .venv/bin/opfusion-train-one-design; do
    [[ -x "$executable" ]] || { echo "missing executable: $executable" >&2; exit 64; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 65; }
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
mkdir -p logs "$BASE_DIR"

if [[ "$SKIP_BASE" == false && ! -f "$BASE_SOURCE/complete.json" ]]; then
    echo "Training shared base.common once..."
    .venv/bin/opfusion-train-one-design \
        --config "$BASE_CONFIG" --job base.common --seed 0 \
        2>&1 | tee "logs/d4_base_$(date -u +%Y%m%dT%H%M%SZ).log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
    mkdir -p "$BASE_OUTPUT_DIR/seed_0"
    rm -rf "$BASE_SOURCE"
    cp -a "$BASE_DIR/sum_a/seed_0/base_common" "$BASE_SOURCE"
fi

[[ -f "$BASE_SOURCE/selected.pt" && -f "$BASE_SOURCE/complete.json" ]] || {
    echo "shared Base is missing; run without --skip-base" >&2
    exit 66
}

if [[ "$BASE_ONLY" == true ]]; then
    echo "Shared Base ready: $BASE_SOURCE"
    exit 0
fi

for cfg in "${CONFIGS[@]}"; do
    job="$(job_for_config "$cfg")"
    name="$(basename "$cfg" .yaml)"
    output="$BASE_DIR/$name"
    complete="$output/seed_0/${job//./_}/complete.json"

    if [[ -f "$complete" ]]; then
        echo "Reusing completed condition: $name"
        continue
    fi

    echo "Staging verified shared Base for $name..."
    .venv/bin/python scripts/stage_d4_shared_base.py \
        --source "$BASE_SOURCE" \
        --destination-output "$output" \
        --base-config "$BASE_CONFIG" \
        --condition-config "$cfg" \
        --seed 0

    echo "Training $name ($job)..."
    .venv/bin/opfusion-train-one-design \
        --config "$cfg" --job "$job" --seed 0 \
        2>&1 | tee "logs/d4_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
    [[ -f "$complete" ]] || { echo "missing completion marker: $complete" >&2; exit 67; }
done

echo "All D4 specialist ablations completed."
