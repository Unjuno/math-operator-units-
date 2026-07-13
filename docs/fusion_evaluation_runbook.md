# Fusion Evaluation Runbook

After a seed finishes, select one generated subset manifest and evaluate it with the canonical surface config.

## All-five matched-joint comparison

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v3/seed_0/fusion_subsets/subset_31.json \
  --split test \
  --examples-per-operator 64 \
  --out evaluations/seed_0_subset_31_test.json
```

For `subset_31`, the report includes:

- base;
- relevant specialist;
- raw bias sum;
- bias mean;
- the exposure-matched all-five joint;
- gold-token NLL;
- generation correctness and exact trace validity;
- Jensen-Shannon divergence and next-token argmax agreement to the joint.

## Intermediate subset diagnostic

```bash
.venv/bin/opfusion-evaluate-fusion \
  --config configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  --manifest runs/gpt_bias_fusion_factory_surface_v3/seed_0/fusion_subsets/subset_03.json \
  --split test \
  --examples-per-operator 64 \
  --out evaluations/seed_0_subset_03_test.json
```

Intermediate manifests deliberately have `joint_reference_checkpoint: null`. Their results are valid for leakage, interference, stability, and task-accuracy diagnostics, but not for claims of equivalence to joint training.

## Alpha

The primary raw condition uses `--alpha 1.0`. A global alpha may be selected on the validation split and then frozen for test evaluation:

```bash
--split validation --alpha 0.5
```

Do not tune alpha on the test split. Input-dependent alpha, routing, and learned correction are separate experiments.

## Checkpoint trajectories

To evaluate a checkpoint observation rather than final weights, use the corresponding manifest under:

```text
seed_<n>/fusion_checkpoint_grid/step_<step>/subset_<mask>.json
```

All compared models in a checkpoint-grid manifest share the same specialist/joint optimizer-step index. The common base remains the completed parent checkpoint used to initialize those models.
