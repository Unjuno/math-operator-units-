# Experiment Configurations

## Canonical production

```text
gpt_bias_fusion_factory_surface_v3.yaml
```

Use only with:

```bash
bash scripts/run_bias_fusion_factory_surface_v3.sh \
  configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml \
  detach
```

This is the primary shared-prefix, ordinary-equality, EOS-terminated experiment.

## Canonical smoke test

```text
gpt_bias_fusion_factory_surface_v3_smoke.yaml
```

This is run automatically by the production launcher. It is not a scientific result and its checkpoints must not be mixed with production outputs.

## Typed-token diagnostic ablation

```text
gpt_bias_fusion_factory_v2.yaml
gpt_bias_fusion_factory_v2_smoke.yaml
```

These profiles predict `<EQ_STEP>` and `<TRACE_STOP>` output classes. They are retained for compatibility and diagnostic comparison only. Their launcher requires:

```bash
OPFUSION_ALLOW_TYPED_V2=1
```

Do not compare or combine typed-v2 checkpoints with surface-v3 checkpoints. Tokenizer profiles and output policies differ.

## Legacy configurations

Other experiment configs in this directory predate the canonical surface-v3 protocol. They may be useful for historical reproduction, but they are not accepted by the production runbook and are not a substitute for the primary condition.

The complete methodological contract is in [`docs/experiment_protocol.md`](../../docs/experiment_protocol.md).
