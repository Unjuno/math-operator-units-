# Operator Applicability Constraints

This project should not treat every named operation as automatically suitable for operator-unit fusion.

A candidate operator is useful only if it can be represented, trained, evaluated, or verified under clear rules.

## 1. Core correction

The operator backlog may be broad, but the applicable operator set is narrower.

```text
candidate operator catalog ≠ trainable operator-unit catalog
```

A source catalog entry can be listed as a candidate, but it should become a learned unit only if the project has a way to generate supervision, extract a stable rule, or verify outputs.

## 2. Applicability requirement

An operator is applicable to this architecture if at least one of the following is true.

### 2.1 Program-generatable exact rule

The operator has an exact evaluator or generator.

Examples:

```text
ADD
NEG
MUL
MIN
MAX
SORT
GCD
MATRIX_TRANSPOSE
DOT
```

These can be trained from synthetic data and checked exactly.

### 2.2 Numeric rule with tolerance

The operator has a stable numeric evaluator with clear tolerances.

Examples:

```text
SQRT
SIN
COS
SVD
EIGEN
INTEGRAL_APPROX
PDE_RESIDUAL
```

These require error tolerance, domain restrictions, and numerical stability tests.

### 2.3 Verifier-backed rule

The operator's output can be checked by a verifier even if generation is hard.

Examples:

```text
candidate proof step
candidate PDE solution
candidate invariant
candidate algorithm
candidate rewrite
```

In this case the model is a candidate generator, not a correctness oracle.

### 2.4 Dataset-extractable regularity

The operator has no exact synthetic generator, but a stable pattern can be learned from curated data.

Examples:

```text
semantic similarity
translation
classification
retrieval relevance
style shift
```

These should usually be marked as `non_math_semantic`, `tool`, `spec_only`, or `dataset_backed`, not as `math_exact`.

### 2.5 Program-only composition

The operator is best represented as a program over existing primitives.

Examples:

```text
SUB = ADD(x, NEG(y))
MEAN = DIV(SUM(x), LEN(x))
STANDARDIZE = DIV(CENTER(x), STD(x))
```

These should not become learned units unless distillation is justified.

## 3. Non-applicable or risky operators

A candidate should not become a learned unit when:

```text
- no evaluator exists
- no verifier exists
- no stable data distribution exists
- the type signature is unclear
- outputs are subjective without a scoring rule
- correctness depends on external mutable state
- the operation is too broad or underspecified
```

Such entries may remain as:

```text
spec_only
tool-backed
fallback spelling
reserved slot candidate
future research
```

## 4. Acceptance levels

Use these levels for backlog entries.

```text
L0 candidate:
  name collected from a source catalog

L1 specified:
  canonical token, kind, domain, and type signature exist

L2 evaluable:
  exact evaluator, numeric evaluator, verifier, or dataset scoring rule exists

L3 trainable:
  train/validation/test distributions are defined

L4 fusible:
  unit + corrector + metrics exist and can be used in runtime fusion
```

Only L4 entries should be used as active fusion units.

## 5. Required metadata

Every operator candidate should eventually declare:

```yaml
applicability:
  level: L0 | L1 | L2 | L3 | L4
  supervision_source: exact_generator | numeric_evaluator | verifier | dataset | program | tool | none
  evaluator_fn: null
  verifier_fn: null
  train_distribution: null
  negative_distribution: null
  domain_restrictions: []
  failure_modes: []
```

## 6. Important distinction

```text
Search broadly.
List candidates broadly.
Accept applicability narrowly.
Train only when evaluable.
Fuse only when trained and corrected.
```

## 7. Final rule

```text
No rule, no evaluator, no verifier, no dataset regularity -> no learned unit.
```
