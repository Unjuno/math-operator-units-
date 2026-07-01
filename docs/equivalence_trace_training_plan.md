# Equivalence Trace Training Plan

Exact evaluators should emit more than final answers, but equality-chain training must not create a shortcut where the model learns that an equals sign should be followed immediately by the final answer and end-of-sequence.

The goal is not:

```text
expression = final_answer <EOS>
```

The goal is:

```text
expression -> valid next transformation candidates -> verified progress -> optional final value
```

## 1. Core distinction

```text
value evaluator:
  computes the final canonical value

equivalence trace generator:
  emits verified equivalent expressions and rewrite steps

trace policy:
  controls when final values and stop tokens are exposed during training
```

All three are needed.

## 2. Main training hazard

Do not train every sequence as a full equality chain ending in the final answer.

Bad global pattern:

```text
1 + 2 + 3 = 3 + 3 = 6 <EOS>
17 * 19 = 323 <EOS>
(x + 3)^2 - 4 = (x + 1)(x + 5) <EOS>
```

If this pattern dominates, the model can learn:

```text
after "=" -> emit final answer -> emit <EOS>
```

That shortcut is harmful because it bypasses search for better transformations, non-monotonic expansions, alternative solution paths, and compressed traces.

## 3. Required anti-shortcut policy

Training data must separate these objectives.

```text
final-value prediction:
  input expression -> canonical value

next-step prediction:
  current expression -> one valid next equivalent expression

rewrite-rule prediction:
  current expression -> applicable rewrite rule

trace-continuation prediction:
  partial trace -> continue with a valid nonterminal step

verification:
  before/after pair -> valid or invalid
```

Do not collapse all objectives into one sequence format.

## 4. Token policy

Use typed delimiters instead of treating plain equals as a universal final-answer marker.

Recommended tokens:

```text
<EQ_STEP>:
  nonterminal equivalence step

<CANONICAL_VALUE>:
  explicit final value field

<TRACE_CONTINUE>:
  request another step

<TRACE_STOP>:
  explicit trace termination decision

<VERIFY_STEP>:
  verify a proposed before/after pair
```

Plain `=` may appear in rendered text, but the model-facing target should distinguish nonterminal equality from final answer emission.

## 5. Target formats

For each example, generate multiple tasks, not one universal chain.

### 5.1 Value task

```yaml
task: final_value
input: "1 + 2 + 3"
target: "6"
```

### 5.2 Next-step task

```yaml
task: next_step
input: "1 + 2 + 3"
target:
  rule: fold_left_add
  after: "3 + 3"
terminal: false
```

### 5.3 Trace-continuation task

```yaml
task: trace_continue
input_trace:
  - "1 + 2 + 3"
  - "3 + 3"
target:
  rule: fold_add
  after: "6"
terminal: true
```

### 5.4 Verification task

```yaml
task: verify_step
before: "1 + 2 + 3"
after: "3 + 3"
target: valid
```

### 5.5 Strategy task

```yaml
task: choose_strategy
input: "17 * 19"
allowed_strategies:
  - direct_multiply
  - difference_of_squares
  - distribute
target: difference_of_squares
```

## 6. Equality chains are allowed, but not everywhere

Rendered equality chains are still useful.

Example:

```text
1 + 2 + 3 = 3 + 3 = 6
```

But they should be used as one training view among several, not as the universal target format.

Rules:

```text
- Do not always terminate after the first equals sign.
- Do not always expose the final answer after equals.
- Do not always include <EOS> immediately after the canonical value.
- Include nonterminal equality steps.
- Include tasks where the final answer is hidden and only the next step is trained.
- Include verification-only tasks where no answer is emitted.
```

## 7. Canonical trace vs augmented traces

Use two trace modes.

### 7.1 Canonical trace

Deterministic trace used for tests and regression.

Example:

```text
1 + 2 + 3 = 3 + 3 = 6
```

Canonical traces should be stored structurally and rendered only when needed.

### 7.2 Augmented traces

Multiple valid traces used for training diversity.

Examples:

```text
1 + 2 + 3 = 1 + 5 = 6
1 + 2 + 3 = 3 + 3 = 6
1 + 2 + 3 = 6
```

All augmented traces must verify to the same canonical value, but some training records should stop before the final answer and ask only for the next verified transformation.

## 8. Non-monotonic trace policy

Equivalence traces need not decrease expression size at every step.

Some useful traces expand before they contract:

```text
17 * 19
= (18 - 1)(18 + 1)
= 18^2 - 1
= 323
```

Therefore:

```text
local expression size may increase
global verified progress must improve
```

Do not train the model to always simplify immediately. Train it to choose verified transformations that improve the solution path.

## 9. Trace validity rule

Do not emit arbitrary or unverified chains.

Even if two expressions have the same value, the rewrite rule should be represented when there is an intermediate step.

Preferred schema:

```yaml
rule: regroup_constants
before: "1 + 2 + 3"
after: "2 + 4"
verified: true
terminal: false
```

The rule must be explicit and verifiable.

## 10. Supported first operators

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

## 11. Training mix

Use a mixed objective distribution.

Recommended initial mix:

```text
final_value: 20%
next_step: 35%
rewrite_rule: 15%
trace_continue: 15%
verify_step: 15%
```

The final-answer objective should not dominate early training.

## 12. Metrics

Add trace-specific and shortcut-specific metrics:

```text
final_value_accuracy
step_validity_rate
chain_validity_rate
canonical_value_agreement
invalid_equivalence_rate
trace_length_ood_accuracy
first_equals_final_answer_rate
equals_to_eos_rate
nonterminal_step_accuracy
strategy_diversity
verified_progress_after_expansion_rate
dead_expansion_rate
```

Especially monitor:

```text
first_equals_final_answer_rate
equals_to_eos_rate
```

These detect whether the model has learned the unwanted shortcut.

## 13. Final rule

```text
Do not train equality as a final-answer shortcut.
Train value separately.
Train next-step transformations separately.
Train verification separately.
Expose full equality chains only as one view, not as the universal target.
```
