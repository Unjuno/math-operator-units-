# Equivalence Trace Training Plan

Exact evaluators should emit more than final answers, but equality-chain training must avoid two opposite failure modes.

Failure mode A:

```text
expression = final_answer <EOS>
```

The model treats `=` as a final-answer trigger.

Failure mode B:

```text
expression = expression' = expression'' = expression''' = ...
```

The model treats `=` as an unconditional continuation trigger and keeps producing equality steps without convergence.

The goal is:

```text
expression -> valid next transformation candidates -> verified progress -> verifier-backed stop
```

## 1. Core distinction

```text
value evaluator:
  computes the final canonical value

equivalence trace generator:
  emits verified equivalent expressions and rewrite steps

trace policy:
  controls when final values, continuation steps, and stop tokens are exposed during training

stop verifier:
  decides whether the current state is an acceptable terminal form

loop detector:
  rejects repeated, cyclic, or no-progress equality chains
```

All five are needed.

## 2. Main training hazards

Do not train every sequence as a full equality chain ending in the final answer.

Bad final-answer pattern:

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

Also do not train equality growth as an unconditional action.

Bad continuation pattern:

```text
after a valid step -> always emit another equality step
```

If this pattern dominates, the model can learn:

```text
after valid state -> emit "=" forever
```

That loop is harmful because it rewards trace length instead of verified progress.

## 3. Required anti-shortcut and anti-loop policy

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

stop prediction:
  partial trace -> continue or stop

verification:
  before/after pair -> valid or invalid

progress scoring:
  partial trace -> progress score or no-progress flag
```

Do not collapse all objectives into one sequence format.

## 4. Token policy

Use typed delimiters instead of treating plain equals as a universal final-answer or universal-continuation marker.

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

<PROGRESS>:
  mark verified progress

<NO_PROGRESS>:
  mark a valid but unhelpful or cyclic step
```

Plain `=` may appear in rendered text, but the model-facing target should distinguish nonterminal equality, final answer emission, and stop decisions.

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

### 5.4 Stop task

```yaml
task: stop_decision
input_trace:
  - "1 + 2 + 3"
  - "3 + 3"
  - "6"
target: stop
reason: canonical_value_reached
```

### 5.5 Verification task

```yaml
task: verify_step
before: "1 + 2 + 3"
after: "3 + 3"
target: valid
```

### 5.6 Progress task

```yaml
task: progress_score
input_trace:
  - "1 + 2 + 3"
  - "3 + 3"
target:
  progress: true
  reason: reduced_unresolved_additions
```

### 5.7 Strategy task

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
- Do not always continue after a valid equality step.
- Include nonterminal equality steps.
- Include stop-decision examples.
- Include no-progress and loop-negative examples.
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

## 9. Anti-loop policy

A valid equivalence step is not automatically a good training target.

Examples of bad but potentially valid loops:

```text
a + b = b + a = a + b = b + a = ...
```

```text
x = x + 0 = x + 0 + 0 = x + 0 + 0 + 0 = ...
```

```text
x = 1 * x = 1 * 1 * x = 1 * 1 * 1 * x = ...
```

These preserve value but do not make useful progress.

Rules:

```text
- Every accepted nonterminal step must be verified equivalent.
- Verification alone is insufficient.
- The step must either improve a progress potential, enable a later contraction, or be explicitly marked as exploratory.
- Repeated canonical forms are rejected.
- Repeated rule cycles are penalized.
- Identity insertions need a bounded budget.
- Trace length has a hard maximum.
- Stop decisions are trained explicitly.
```

## 10. Progress potential

Use a progress potential to distinguish useful growth from equality spam.

Candidate fields:

```text
token_length
tree_depth
unresolved_operator_count
numeric_difficulty
canonical_distance
new_structure_exposed
future_contraction_available
```

The potential does not need to decrease at every step, but over a bounded window it must improve.

```text
local non-monotonicity allowed
bounded-window no-progress forbidden
```

## 11. Trace validity rule

Do not emit arbitrary or unverified chains.

Even if two expressions have the same value, the rewrite rule should be represented when there is an intermediate step.

Preferred schema:

```yaml
rule: regroup_constants
before: "1 + 2 + 3"
after: "2 + 4"
verified: true
terminal: false
progress: true
```

The rule must be explicit and verifiable.

## 12. Supported first operators

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

## 13. Training mix

Use a mixed objective distribution.

Recommended initial mix:

```text
final_value: 15%
next_step: 30%
rewrite_rule: 15%
trace_continue: 15%
stop_decision: 10%
verify_step: 10%
progress_score: 5%
```

The final-answer objective should not dominate early training, and the continuation objective should not dominate stop training.

## 14. Metrics

Add trace-specific, shortcut-specific, and loop-specific metrics:

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
equals_continuation_rate
stop_decision_accuracy
repeated_state_rate
rule_cycle_rate
identity_insertion_rate
max_trace_length_violation_rate
no_progress_window_rate
```

Especially monitor:

```text
first_equals_final_answer_rate
equals_to_eos_rate
equals_continuation_rate
repeated_state_rate
no_progress_window_rate
```

These detect whether the model has learned either unwanted shortcut: terminate immediately or continue forever.

## 15. Final rule

```text
Do not train equality as a final-answer shortcut.
Do not train equality as an unconditional continuation shortcut.
Train value separately.
Train next-step transformations separately.
Train stop decisions separately.
Train verification and progress separately.
Expose full equality chains only as one view, not as the universal target.
```
