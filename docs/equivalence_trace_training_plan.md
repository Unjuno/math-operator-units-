# Equivalence Trace Training Plan

Exact evaluators should emit more than final answers.

For many mathematical operators, the training target should include machine-verifiable equivalent forms and equality chains.

Example:

```text
1 + 2 + 3 = 3 + 3 = 6
```

The purpose is to teach compositional equivalence rather than only final answer mapping.

## 1. Core distinction

```text
value evaluator:
  computes the final canonical value

equivalence trace generator:
  emits a verified sequence of equivalent expressions
```

Both are needed.

## 2. Output schema

A trace target should be represented structurally, not only as a string.

```yaml
operator_id: scalar.add
input_expr: "1 + 2 + 3"
canonical_value: 6
equivalent_forms:
  - "1 + 2 + 3"
  - "3 + 3"
  - "6"
steps:
  - rule: fold_left_add
    before: "1 + 2 + 3"
    after: "3 + 3"
  - rule: fold_add
    before: "3 + 3"
    after: "6"
verified: true
```

Every adjacent pair in the chain must be checked by the evaluator or verifier.

## 3. Why final answers alone are weak

If training only sees:

```text
1 + 2 + 3 -> 6
```

then the model may learn shallow input-output associations.

If it sees:

```text
1 + 2 + 3 = 3 + 3 = 6
```

then the model has a supervised path through equivalent states.

This supports:

```text
- compositional arithmetic
- program rewriting
- intermediate checking
- candidate verification
- less reliance on final-only shortcuts
```

## 4. Canonical trace vs augmented traces

Use two trace modes.

### 4.1 Canonical trace

Deterministic trace used for tests and regression.

Example:

```text
1 + 2 + 3 = 3 + 3 = 6
```

### 4.2 Augmented traces

Multiple valid traces used for training diversity.

Examples:

```text
1 + 2 + 3 = 1 + 5 = 6
1 + 2 + 3 = 3 + 3 = 6
1 + 2 + 3 = 6
```

All augmented traces must verify to the same canonical value.

## 5. Trace validity rule

Do not emit arbitrary or unverified chains.

Even if two expressions have the same value, the rewrite rule should be represented when there is an intermediate step.

Preferred schema:

```yaml
rule: regroup_constants
before: "1 + 2 + 3"
after: "2 + 4"
```

The rule must be explicit and verifiable.

## 6. Supported first operators

Start with exact finite operators:

```text
scalar.add
scalar.neg
scalar.abs
scalar.pos
scalar.min
scalar.max
aggregation.sum
aggregation.center
bias.add
bias.sub
bias.center
bias.pos
```

Later extend to:

```text
scalar.mul
aggregation.prod
linalg.dot
linalg.matmul
program-only derived operators
symbolic rewrites
```

## 7. Training targets

For each example, generate several target formats.

```text
final_value:
  6

equality_chain:
  1 + 2 + 3 = 3 + 3 = 6

structured_trace:
  list of verified rewrite steps
```

This allows ablation:

```text
final-only vs equality-chain vs structured-trace
```

## 8. Metrics

Add trace-specific metrics:

```text
final_value_accuracy
step_validity_rate
chain_validity_rate
canonical_value_agreement
invalid_equivalence_rate
trace_length_ood_accuracy
```

## 9. Final rule

```text
Train value.
Train equivalence.
Verify every step.
Do not reward unverified chains.
```
