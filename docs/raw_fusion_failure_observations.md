# Raw Fusion Failure Observations

## 1. Status of this note

This document records qualitative observations from early small proxy experiments based on perceptron, NN, or MLP models.

These observations are not GPT results and do not establish a final fusion rule. They are retained because they identify failure modes that the GPT experiment should measure directly.

The note must not be read as proof that every operator model requires a corrector.

## 2. Initial expectation

A simple expectation was:

```text
specialist model on irrelevant input -> approximately neutral bias
```

If this held, raw fusion could be attempted by direct addition or averaging:

```text
z_raw(x) = z_base(x) + sum_k B_k(x)
```

where:

```text
B_k(x) = z_k(x) - z_base(x)
```

The intended behavior was that relevant specialists would contribute strongly while irrelevant specialists would remain close to zero.

## 3. Preliminary proxy observation

In the early proxy experiments, irrelevant models sometimes produced sharp, structured, and wrong outputs instead of neutral outputs.

A representative qualitative pattern was:

```text
addition-only proxy
+ subtraction-like or otherwise unknown input
-> unknown operation assimilated into addition-like behavior
```

This pattern was called:

```text
operator assimilation error
```

Japanese:

```text
未知演算子の既知演算子への同化エラー
```

Because the experiments were small and used non-GPT architectures, this should be treated as a hypothesis to reproduce, not as an established property of the current GPT models.

## 4. Why the observation matters

Bias fusion can amplify an error when several irrelevant fields support the same wrong token.

For an incorrect token `v_wrong`:

```text
B_1(v_wrong | x) > 0
B_2(v_wrong | x) > 0
B_3(v_wrong | x) > 0
```

can produce:

```text
sum_k B_k(v_wrong | x) >> 0
```

The problem is therefore not only specialist accuracy. The experiment must also measure whether inactive or out-of-domain specialists create non-neutral contributions.

## 5. GPT hypotheses to test

The GPT experiment should test the following competing hypotheses.

### H1: approximate neutrality

```text
irrelevant GPT specialist -> small or diffuse bias field
```

If supported, simple raw fusion may be sufficient in some conditions.

### H2: structured leakage

```text
irrelevant GPT specialist -> structured non-zero field
```

If supported, raw fusion may amplify incorrect tokens.

### H3: scale mismatch

```text
specialist fields contain useful directions
but their magnitudes are not directly comparable
```

If supported, fixed normalization or coefficient tuning may be sufficient without routing.

### H4: applicability dependence

```text
specialist contribution is useful only on a subset of prefixes
```

If supported, oracle or learned weighting may improve fusion.

### H5: common-component duplication

```text
fields measured from random initialization repeat shared syntax or format learning
```

If supported, a trained common base is required before extracting operator-specific bias.

## 6. Required raw-fusion measurements

For each specialist, subset, checkpoint, and seed, record:

```text
inactive_bias_l2
inactive_centered_bias_l2
inactive_top_probability
inactive_entropy
unknown_operator_assimilation_rate
wrong_token_amplification
raw_fusion_task_loss
raw_fusion_exact_accuracy
raw_fusion_trace_accuracy
joint_vs_fused_kl_or_jsd
field_agreement
field_conflict
value_ood_accuracy
length_ood_accuracy
```

The experiment should retain token-level examples of the largest amplification events for manual inspection.

## 7. Length OOD observation

The early proxy work also suggested a length breakpoint: models trained on short expressions degraded after the sequence exceeded the training range.

Useful measurements are:

```text
L_train:
  maximum trained expression length

L_break:
  first evaluated length where accuracy falls below a fixed threshold

length_margin:
  L_break - L_train
```

A larger `length_margin` is not evidence of unbounded algorithmic generalization.

The GPT experiment must use explicit IID and length-OOD splits rather than inferring generalization from one validation distribution.

## 8. Correction methods are later comparisons

If raw fusion shows error amplification, possible follow-up methods include:

- fixed averaging;
- coefficient sweeps;
- centered fields;
- norm balancing;
- oracle applicability weights;
- learned scalar weights;
- token-wise weighting;
- routing;
- projection removal;
- learned residual correction.

No method should be built into the definition of an operator model before the raw baseline is measured.

A corrector is one hypothesis among several. It is not a conclusion of the proxy observation.

## 9. External-facing statement

```text
Early small proxy experiments suggested that models outside their trained operator domain may emit structured, confident, and incorrect biases rather than neutral outputs. If the same behavior occurs in GPT specialists, direct bias fusion could amplify incorrect tokens. The current GPT experiment therefore measures inactive leakage and error amplification before comparing optional weighting, routing, or correction methods.
```

Japanese:

```text
初期の小規模proxy実験では、学習対象外の演算入力に対して、モデルが中立なbiasではなく、
構造化された尖った誤biasを出す可能性が示唆された。同じ現象がGPT specialistでも起きるなら、
単純なbias fusionによって誤tokenが増幅される。そこでGPT実験では、まずraw fusion、
inactive leakage、誤差増幅を測定し、その後にweighting、routing、補正を比較する。
```

## 10. Final experimental rule

```text
Do not assume that irrelevant specialists are neutral.
Do not assume that a corrector is necessary.
Measure raw bias fields and their interactions first.
Use the result to determine which, if any, correction mechanism is justified.
```