# Shared Numeric and Equality ABI

Numbers and equality are shared infrastructure across all operator units.

They are not owned by `ADD`, `MUL`, `NEG`, `VERIFY`, `STOP`, or any other individual operator.

## 1. Core rule

```text
numbers and equality are shared ABI tokens
operator units are learned transformations over that shared ABI
```

The model family must preserve a common interpretation of:

```text
numeric tokens:
  <N_0>, <N_1>, <N_2>, ...

equality token:
  <EQ> or rendered "="

structural tokens:
  parentheses, separators, task markers, trace markers
```

Operator-specific units must not redefine these tokens.

## 2. Why this matters

All units emit logits, bias fields, candidate expressions, or verification judgments in the same token/output space.

If different units assign incompatible semantics to numeric tokens or equality, fusion becomes ill-defined.

Example failure:

```text
ADD unit:
  "=" means next addition rewrite

VERIFY unit:
  "=" means comparison target

STOP unit:
  "=" means maybe terminate
```

These roles differ, but the underlying equality symbol must remain the same shared relation marker.

## 3. Common vs operator-specific learning

Separate training into shared and operator-specific layers.

### Shared layer

This layer teaches numeric identity, equality syntax, and basic equivalence recognition.

Examples:

```text
<VERIFY_EQ> 6 = 6 -> VALID
<VERIFY_EQ> 3 + 4 = 7 -> VALID
<VERIFY_EQ> 3 + 4 = 8 -> INVALID
<PROGRESS?> 6 = 6 -> NO_PROGRESS
```

These examples can be shared across models or used to align units to the same ABI.

### Operator-specific layer

This layer teaches transformations valid under a particular operator family.

ADD examples:

```text
<OP_ADD> <NEXT_EQ> 3 + 4 + 5 -> 7 + 5
<OP_ADD> <NEXT_EQ> 3 + 4 + 5 -> 3 + 9
<OP_ADD> <NEXT_EQ> 7 + 5 -> 12
```

MUL examples:

```text
<OP_MUL> <NEXT_EQ> 3 * 4 * 5 -> 12 * 5
<OP_MUL> <NEXT_EQ> 12 * 5 -> 60
```

The numbers and equality marker are shared; the transformation distribution is operator-specific.

## 4. Equality is not a single training objective

The equality marker must not be trained as either:

```text
= final_answer <EOS>
```

or:

```text
= next = next = next forever
```

Instead, it should indicate an equivalence edge whose usefulness is determined by external or learned tasks:

```text
VERIFY_EQ:
  before/after are equivalent or not

PROGRESS:
  the step helps the solution path or not

STOP:
  the current trace is acceptable to terminate or not

NEXT_EQ:
  propose a next equivalent expression
```

## 5. Training distribution rule

Do not train each operator only on final answers.

For an operator such as ADD, train on a biased distribution over valid transformations:

```text
progressive intermediate rewrites
terminal/canonical rewrites
reordering and associativity rewrites
bounded expansions
no-progress equivalences as labeled negatives for progress
invalid equivalences for verification
```

The data distribution is intentionally designed.

The objective is not to approximate a uniform distribution over all equivalent expressions. The objective is to learn a useful, verifier-compatible transformation distribution.

## 6. Valid equivalence is not enough

A step can be valid but useless.

Examples:

```text
6 = 6
x = x + 0
x = 1 * x
```

These should be accepted by equivalence verification but usually rejected or downweighted by progress training.

Required distinction:

```text
VALID_EQ:
  the equality is true

PROGRESS_EQ:
  the equality step helps solve, simplify, expose structure, or reach a target

NO_PROGRESS_EQ:
  the equality is true but does not usefully advance the trace
```

## 7. Fusion implication

For fusion, all unit checkpoints must declare the same:

```text
tokenizer_profile
vocab_hash
output_space_id
numeric_token_policy
equality_token_policy
```

The equality and number tokens are part of the model ABI. A unit trained with incompatible numeric or equality semantics must not be fused with other units.

## 8. Final rule

```text
Shared ABI:
  numbers, equality, structure

Operator unit:
  transformation distribution over the shared ABI

Verifier/progress/stop units:
  judgments over traces expressed in the shared ABI
```

In Japanese:

```text
数字と等号は全unit共通のABI。
各operator unitは、その共通ABI上の変形分布を学習する。
```
