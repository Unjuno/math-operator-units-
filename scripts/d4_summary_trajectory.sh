#!/bin/bash
# D4 Trajectory Summary - Extract key metrics from trajectory evaluations

set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")"
cd "$REPO_ROOT"

EVAL_DIR="evaluations/d4_specialist_ablation"

if [[ ! -d "$EVAL_DIR" ]]; then
    echo "No evaluations found in $EVAL_DIR"
    exit 1
fi

echo "=== D4 Specialist Ablation Trajectory Summary ==="
echo ""

for op in aggregation.sum scalar.neg; do
    echo ">>> $op <<<"
    echo ""
    
    # Find all evaluation files
    files=($(find "$EVAL_DIR" -name "${op//./_}_step_*.json" | sort -V))
    
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "  No evaluations found"
        echo ""
        continue
    fi
    
    printf "%-8s %10s %10s %10s %10s %10s %10s\n" "Step" "TraceVal" "FinalAcc" "EOSAcc" "TF_TokAcc" "TF_SeqAcc" "TF_First"
    printf "%-8s %10s %10s %10s %10s %10s %10s\n" "----" "--------" "--------" "------" "---------" "---------" "-------"
    
    for f in "${files[@]}"; do
        step=$(basename "$f" .json | sed 's/.*step_//')
        
        trace_valid=$(jq -r ".operators[\"$op\"].generation.trace_validity // 0" "$f" 2>/dev/null)
        final_acc=$(jq -r ".operators[\"$op\"].generation.final_value_accuracy // 0" "$f" 2>/dev/null)
        eos_acc=$(jq -r ".operators[\"$op\"].generation.stop_accuracy // 0" "$f" 2>/dev/null)
        tf_tok=$(jq -r ".operators[\"$op\"].teacher_forced.token_accuracy // 0" "$f" 2>/dev/null)
        tf_seq=$(jq -r ".operators[\"$op\"].teacher_forced.sequence_argmax_accuracy // 0" "$f" 2>/dev/null)
        tf_first=$(jq -r ".operators[\"$op\"].teacher_forced.first_token_accuracy // 0" "$f" 2>/dev/null)
        
        printf "%-8s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n" \
            "$step" "$trace_valid" "$final_acc" "$eos_acc" "$tf_tok" "$tf_seq" "$tf_first"
    done
    echo ""
done