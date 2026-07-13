# Optional Reliability-Calibrator Training Plan

> **Status:** deferred ablation. Do not train a calibrator until the GPT specialists, common base, joint reference, raw fusion baseline, and error-amplification measurements are complete.

## 1. Purpose

This document describes one possible follow-up method when raw bias fusion shows structured inactive leakage or wrong-token amplification.

It does not define the core model factory. It does not imply that every specialist requires a paired corrector. It must be compared against simpler alternatives such as fixed coefficients, averaging, centering, and norm balancing.

## 2. Preconditions

A calibrator experiment is valid only after the following artifacts exist:

1. a trained common `BaseGPT`;
2. frozen specialist checkpoints branched from that base;
3. a frozen joint reference checkpoint;
4. executable raw fusion over the same prefix and vocabulary;
5. exact task and trace verifiers;
6. measured inactive leakage and amplification events;
7. aligned IID and OOD evaluation splits.

The calibrator must not be used to hide defects in the base model, specialist training, prompt masking, tokenizer ABI, or exposure matching.

## 3. Candidate formulation

For specialist `M_k`, compute:

```text
B_k(v | x) = z_k(v | x) - z_base(v | x)
```

A reliability model may predict:

### Scalar weight

```text
R_k(x, summary(B_k)) -> r_k in [0, 1]
B_tilde_k = r_k B_k
```

### Token-wise weight

```text
R_k(x, B_k) -> r_k(v | x) in [0, 1]
B_tilde_k(v | x) = r_k(v | x) B_k(v | x)
```

### Removal field

```text
R_k(x, B_k) -> E_k(v | x)
B_tilde_k = B_k - E_k
```

These variants have different capacity and must not be grouped as one method.

## 4. Training order

```text
1. Train and freeze BaseGPT.
2. Train and freeze specialist M_k.
3. Run M_k on owned, non-owned, mixed, and OOD prefixes.
4. Save the actual emitted bias fields and outcomes.
5. Label reliability using exact task and trace verifiers.
6. Train R_k without updating M_k.
7. Evaluate raw and calibrated fusion on the same held-out prefixes.
```

The generator must not self-certify its own output.

## 5. Inputs

Candidate inputs include:

```text
prefix x
bias field B_k
centered field norm
top-token identities and margins
entropy and top probability
agreement or conflict with other fields
exact verifier result
trace-progress result
value and length OOD indicators
```

A critical ablation removes explicit operator identity. This determines whether the model detects reliability from the generated field or merely routes by an operator token.

## 6. Data families

### Owned cases

- in-domain operator prompts;
- valid generated traces;
- correct final answers;
- stable fields across equivalent states.

### Non-owned cases

- prompts from other operator families;
- mixed-operator prompts;
- irrelevant specialists evaluated on valid foreign tasks.

### OOD cases

- values outside the training range;
- longer traces;
- malformed expressions;
- rare token combinations;
- premature or repeated stop/equality tokens.

### Hard negatives

- confident wrong outputs;
- low-entropy wrong fields;
- operator-assimilation events;
- fields that strongly amplify an incorrect token when fused.

## 7. Targets

Possible targets are:

```text
exact correctness
complete-trace validity
verifier acceptance
wrong-token amplification
joint-reference divergence
oracle usefulness of including the field
```

A scalar target may be derived from held-out measured utility, but the formula and threshold must be fixed before evaluating the calibrator.

The label must not be based solely on whether the operator token matches the specialist name.

## 8. Losses

Candidate losses include:

```text
L_reliability:
  BCE or regression against measured reliability

L_effect:
  KL or JSD between the calibrated fused distribution and the selected reference

L_preserve:
  penalty for suppressing a correct specialist contribution

L_amplify:
  penalty for retaining a field that increases wrong-token probability
```

A high-capacity removal-field model must be evaluated for answer leakage. It may learn the target distribution directly instead of calibrating the specialist.

## 9. Required baselines

Every result must compare:

1. base only;
2. each specialist alone;
3. raw sum;
4. arithmetic mean;
5. fixed coefficient sweep;
6. centered-field fusion;
7. norm-balanced fusion;
8. oracle applicability;
9. learned scalar reliability;
10. token-wise or removal-field variants only if justified.

A calibrator is useful only when it improves over simple non-learned baselines and preserves valid competing contributions.

## 10. Metrics

```text
raw_vs_calibrated_task_loss
raw_vs_calibrated_exact_accuracy
raw_vs_calibrated_trace_accuracy
joint_vs_fused_kl_or_jsd
wrong_token_amplification_reduction
correct_contribution_preservation
false_attenuation_rate
inactive_leakage_reduction
reliability_calibration_error
operator_identity_ablation_delta
```

Report results by operator, subset, seed, checkpoint, and OOD split.

## 11. Stop conditions

Do not continue to more expressive calibrators when:

- raw fusion already works adequately;
- fixed scaling solves the measured problem;
- the calibrator only reproduces operator-token routing;
- task accuracy improves while joint matching or trace validity degrades;
- the calibrator predicts the answer independently of specialist fields;
- improvements disappear under OOD evaluation.

## 12. Short framing

```text
Reliability calibration is an optional response to measured error amplification. It is trained only after the base, specialists, joint reference, and raw fusion baseline are fixed. Its purpose is to test whether learned weighting improves the same fusion problem, not to redefine bias fusion or assume in advance that every specialist requires correction.
```

Japanese:

```text
信頼性補正は、raw bias fusionで誤差増幅が実測された場合に検討する追加実験である。
共通base、specialist、joint reference、raw baselineを固定した後に学習し、固定係数や中心化などの
単純手法と比較する。補正器はbias fusionの定義ではなく、全specialistに必須とも仮定しない。
```