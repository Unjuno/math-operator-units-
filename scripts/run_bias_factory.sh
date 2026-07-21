#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUNS_DIR="runs/bias_factory"
mkdir -p "$RUNS_DIR" logs

# First generate all configs
.venv/bin/python scripts/generate_bias_factory_configs.py

SIZES=("nano" "small" "medium" "1m")
OPS=("sum" "neg" "add" "min" "max")

_op_job() {
    case "$1" in
        sum) echo "aggregation.sum" ;;
        neg) echo "scalar.neg" ;;
        add) echo "scalar.add" ;;
        min) echo "scalar.min" ;;
        max) echo "scalar.max" ;;
        *) echo "unknown" ;;
    esac
}

echo ""
echo "============================================="
echo "PHASE 1: Joint models  (4 runs)"
echo "============================================="

for size in "${SIZES[@]}"; do
    cfg="configs/experiments/bias_factory/joint_${size}.yaml"
    complete="$RUNS_DIR/$size/seed_0/base_common/complete.json"

    if [[ -f "$complete" ]]; then
        echo "Reusing completed: joint $size"
        continue
    fi

    echo "Training joint $size..."
    .venv/bin/opfusion-train-one-design \
        --config "$cfg" --job base.common --seed 0 \
        2>&1 | tee "logs/bias_factory_joint_${size}_$(date -u +%Y%m%dT%H%M%SZ).log"

    exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        echo "FAILED: joint $size (exit $exit_code)" >&2
        exit 1
    fi
    echo "Completed: joint $size"
done

echo ""
echo "============================================="
echo "PHASE 2: Specialist models  (20 runs)"
echo "============================================="

for size in "${SIZES[@]}"; do
    for op in "${OPS[@]}"; do
        cfg="configs/experiments/bias_factory/spec_${size}_${op}.yaml"
        spec_name="${size}_${op}"
        op_job="$(_op_job "$op")"
        op_dir="${op_job//./_}"
        complete="$RUNS_DIR/spec_${spec_name}/seed_0/${op_dir}/complete.json"

        if [[ -f "$complete" ]]; then
            echo "Reusing completed: spec $spec_name"
            continue
        fi

        echo "Training spec $spec_name (job=$op_job)..."
        .venv/bin/opfusion-train-one-design \
            --config "$cfg" --job "$op_job" --seed 0 \
            2>&1 | tee "logs/bias_factory_spec_${spec_name}_$(date -u +%Y%m%dT%H%M%SZ).log"

        exit_code=${PIPESTATUS[0]}
        if [[ $exit_code -ne 0 ]]; then
            echo "FAILED: spec $spec_name (exit $exit_code)" >&2
            # continue with next specialist instead of aborting
        fi
        echo "Completed: spec $spec_name"
    done
done

echo ""
echo "============================================="
echo "Bias Factory complete"
echo "  Joint models:  4"
echo "  Specialists:  20"
echo "  Total:        24 runs"
echo "  Snapshots/run: 5 steps (39/156/635/2559/9800)"
echo "============================================="
