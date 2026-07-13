# GPT Operator Model Factory

## 1. Purpose

The factory exists to create the controlled GPT checkpoints required for a later bias-fusion experiment.

The research target is not the mathematical operators themselves. The operators are used because their training data, intermediate transformations, final answers, and failure cases can be generated and verified exactly.

The final comparison is intended to involve:

- a trained common base GPT;
- independently specialized operator GPTs branched from that base;
- a joint reference GPT trained on the union of operator data;
- runtime bias fusion of the independent specialists;
- measurements of task correctness, distribution matching, inactive leakage, and error amplification.

The factory does not establish that bias fusion works. It produces the checkpoints needed to test that question.

## 2. Current v1 checkpoint set

The current implementation creates the following six jobs per seed:

| Job | Current training distribution | Role |
|---|---|---|
| `scalar.add` | binary addition traces | specialist |
| `aggregation.sum` | variable-length sum contractions | specialist |
| `scalar.neg` | sign inversion traces | specialist |
| `scalar.min` | pairwise minimum contractions | specialist |
| `scalar.max` | pairwise maximum contractions | specialist |
| `joint.all_five` | mixture of all five families | joint reference candidate |

The five specialists define `2^5 = 32` runtime subsets. These are checkpoint manifests, not 32 separately trained models.

## 3. Architecture contract

The current profile is `gpt_operator_1m_v1`:

| Field | Value |
|---|---:|
| Vocabulary size | 2,064 |
| Hidden width | 112 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Feed-forward width | 448 |
| Context length | 128 |
| Weight tying | enabled |
| Parameter count | 848,624 |
| Hard upper bound | 1,000,000 |

The runner constructs the model and refuses training when the actual parameter count exceeds the configured bound.

The profile is intentionally small. A 24 GB RTX 3090 should have ample memory for this architecture, but throughput and batch choices must still be measured on the actual environment.

## 4. Tokenizer contract

`operator_experiment_v1` is a fixed-vocabulary tokenizer containing:

- PAD, BOS, EOS, and UNK;
- explicit equality-step and trace-stop tokens;
- structural tokens for the synthetic expressions;
- atomic integers from `-1024` through `1024`;
- exactly five operator-family tokens.

It contains no blank or reserved operator slots. The ordered vocabulary is hashed and written into every checkpoint.

All models included in one fusion comparison must use the same tokenizer profile, ordered vocabulary, and vocabulary hash.

## 5. Equality-trace construction

The generator creates deterministic contractive traces. A resolved prefix is replaced by its value and is not copied into the next state.

```text
<OP_AGG_SUM> 1 + 2 + 3 + 4
<EQ_STEP> 3 + 3 + 4
<EQ_STEP> 6 + 4
<EQ_STEP> 10
<TRACE_STOP>
```

Minimum and maximum use analogous contractions. Addition and negation currently use a single verified transition.

## 6. Required correction: trained common base

The current v1 factory saves `shared_initial.pt`, which is a shared random initialization. It is not a trained base model.

For operator-specific bias extraction, the intended v2 structure is:

```text
shared random initialization
        ↓
train BaseGPT on shared syntax and trace conventions
        ↓
branch all specialists and joint references from BaseGPT
```

The bias field should then be computed as:

```text
B_k(x) = z_k(x) - z_base(x)
```

Using random initialization as `z_base` would mix common language-format learning with operator-specific learning and could cause repeated common components to be amplified during fusion.

## 7. Required correction: target masking

The current v1 language-model batch labels every non-padding token in the sequence. This includes randomly sampled prompt operands.

A random operand cannot be predicted from the preceding operator token, so the full-sequence cross-entropy has an irreducible loss floor. It cannot be interpreted as a task loss that should approach zero.

The intended training record must separate prompt and response:

```text
prompt:
  <OP_AGG_SUM> 1 + 2 + 3 + 4

response:
  <EQ_STEP> 3 + 3 + 4
  <EQ_STEP> 6 + 4
  <EQ_STEP> 10
  <TRACE_STOP>
```

Prompt labels must be set to `-100`; only response tokens should contribute to the task cross-entropy.

This task loss is distinct from the later fusion-matching loss:

```text
L_task  = CE(correct response tokens, model distribution)
L_match = D(p_joint, p_fused)
```

`L_match` can be zero when distributions match. `L_task` can approach zero only when the target is deterministically recoverable from the visible prompt and previous response tokens.

## 8. Required correction: fair joint reference

At the same optimizer step, each specialist currently receives a full batch from one operator, while `joint.all_five` divides its samples across five operators.

Therefore same-step checkpoints do not have equal per-operator exposure.

The revised experiment must report at least two reference conditions:

1. `joint.step_matched`: same optimizer-step count;
2. `joint.exposure_matched`: same examples or target tokens per operator as the corresponding specialists.

The primary fusion comparison should use explicit per-operator exposure accounting. A large joint batch or gradient accumulation can be used to match exposure on a 24 GB RTX 3090 after a hardware smoke benchmark.

## 9. Checkpoint policy

Permanent checkpoints should include:

- model parameters;
- optimizer state;
- completed training step;
- CPU and CUDA RNG state;
- task loss;
- validation metrics;
- per-operator metrics;
- distance from the common base and random initialization;
- model and tokenizer ABI metadata;
- example and target-token exposure counts.

The current early and periodic checkpoint schedule remains useful:

```text
0, 100, 300, 1,000, 3,000, 10,000, 30,000, 100,000, 200,000
```

However, a 200,000-step production run must not be selected solely because the implementation accepts that value. The actual run length should follow a benchmark and convergence pilot.

## 10. Fusion experiment produced from these checkpoints

For each aligned checkpoint and subset:

```text
B_k(x) = z_k(x) - z_base(x)
B_fused(x) = F(B_k(x) for k in active subset)
z_fused(x) = z_base(x) + B_fused(x)
p_fused(x) = softmax(z_fused(x))
```

Compare against:

```text
p_joint(x) = softmax(z_joint(x))
```

using both distribution and task measurements:

- KL or Jensen-Shannon divergence;
- exact final-answer accuracy;
- complete-trace accuracy;
- equality-step validity;
- stopping accuracy;
- inactive bias norm;
- wrong-token amplification;
- value OOD and length OOD performance.

The existing v1 code writes subset manifests, but it does not yet execute this fusion evaluation.

## 11. Routing and correction status

Routing, scalar weighting, token-wise calibration, confidence weighting, and removal fields are later experimental conditions.

They are not prerequisites for producing specialist checkpoints and are not part of the basic definition of bias fusion.

The required order is:

```text
raw fusion baseline
    ↓
measure leakage and error amplification
    ↓
compare optional routing or correction methods
```

This prevents a router or corrector from being mistaken for evidence that raw specialist fields are composable.

## 12. Current status

The v1 factory is a functioning checkpoint-generation scaffold with:

- a GPT-only architecture;
- fixed tokenizer ABI;
- CUDA execution;
- resumable checkpoints;
- shared random initialization;
- deterministic synthetic data;
- five specialists and one joint job;
- 32 subset manifests.

It is **not yet the approved four-week production experiment** because the trained base, response masking, fair joint exposure, exact generation evaluation, OOD splits, and executable fusion evaluator remain to be implemented.

## 13. Approval criteria for the long run

Before unattended production training:

1. all unit tests pass;
2. target masking is verified with a hand-inspected batch;
3. the trained common base checkpoint is created and recorded;
4. specialist and joint exposure accounting is explicit;
5. exact generation metrics work on a small overfit test;
6. the fusion evaluator runs at least one subset end to end;
7. a 1,000-step RTX 3090 benchmark records throughput, VRAM, and projected duration;
8. checkpoint resume is tested after forced interruption;
9. the production configuration is assigned a new experiment ID.