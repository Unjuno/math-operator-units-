# Runtime Bias Set Swapping

This repository does not require all possible units to be active in every run.

The architecture supports two different concepts:

```text
full registry:
  all known operators and possible units

runtime fusion set:
  the subset of units/biases selected for a specific task, domain, or experiment
```

## 1. Core idea

Fusion is performed over a selected set of units:

```text
z_final = z_0 + Σ_{k in S_runtime} g_k(x) b_k(x)
```

where `S_runtime` is not necessarily the full registry.

This means the system can swap task-specific bias/operator sets without changing the tokenizer or retraining the base model.

## 2. Why this is a strength

A fixed tokenizer and registry allow shared compatibility, but runtime does not need to pay for every possible unit.

For example:

```text
arithmetic task:
  ADD, NEG, MUL, DIV, SIGN, COMPARE

linear algebra task:
  DOT, MATMUL, TRANSPOSE, SOLVE, PROJ

PDE task:
  D_X, D_T, D_XX, LAPLACIAN, BC, IC, RESIDUAL

LLM control task:
  BIAS_ADD, BIAS_REMOVE, BIAS_AGREE, KL_BUDGET, ENTROPY_MATCH
```

Each task can load a different `S_runtime` while preserving the same tokenizer ABI.

## 3. Difference from full always-on fusion

Full always-on fusion:

```text
S_runtime = all units
```

Runtime bias-set fusion:

```text
S_runtime = task-specific subset
```

The second is usually more efficient and safer.

The corrector is still useful because even inside a selected runtime set, not every unit applies to every input.

## 4. Two levels of selection

### 4.1 Runtime set selection

This selects which units are loaded for a task.

Examples:

```text
math_scalar_v1
bias_control_v1
pde_1d_heat_v1
linalg_solver_v1
```

### 4.2 Per-input self-suppression

Inside the runtime set, each corrector decides whether its unit should contribute.

```text
unit loaded ≠ unit active
```

A unit may be loaded but suppressed for a specific input.

## 5. Why this avoids unit explosion at runtime

The registry can contain hundreds or thousands of operators.

The runtime fusion set may contain only 8, 16, 64, or 128 units.

This reduces:

```text
- compute cost
- inactive leakage
- conflict probability
- debugging complexity
```

while preserving long-term extensibility.

## 6. Required metadata

A runtime fusion set should be declared as a manifest:

```yaml
fusion_set_id: pde_1d_heat_v1
tokenizer_profile: tokenizer_core_v1
vocab_hash: sha256:...
registry_assignment_hash: sha256:...
output_space_id: core_logits_v1
units:
  - calc.dx
  - calc.dt
  - calc.dxx
  - calc.laplacian
  - pde.heat_residual
  - pde.bc_enforce
  - pde.ic_enforce
  - verify.residual
```

The manifest must declare compatibility hashes so that incompatible units cannot be silently mixed.

## 7. Metrics

Metrics should be reported per runtime fusion set:

```text
runtime_unit_count
active_unit_count_mean
inactive_leakage_mean
inactive_leakage_p95
conflict_rate
false_neutral_rate
latency
memory
accuracy_or_residual
```

This makes it possible to compare:

```text
8-unit fusion vs 64-unit fusion
math-only fusion vs math+verifier fusion
full always-on vs task-specific fusion
```

## 8. Important distinction

Reserved tokens are for long-term compatibility.

Runtime fusion sets are for practical execution.

```text
reserved tokens solve future ABI growth
runtime fusion sets solve current task specialization
correctors solve per-input applicability
verifiers solve correctness
```

## 9. Final rule

```text
Do not run everything unless the experiment is explicitly testing full always-on scaling.
Use registry-wide compatibility, but runtime-specific fusion sets.
```
