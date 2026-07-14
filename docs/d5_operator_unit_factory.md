# D5 operator-unit factory

D5 isolates initialization from task/data learnability before returning to fusion.
It creates one weak multitask Base and four controlled operator models from seed 0:

| Condition | Operator | Initialization | Training domain |
|---|---|---|---|
| `sum_scratch` | `aggregation.sum` | deterministic random initialization | 3 terms, operands ±8, canonical full trace |
| `sum_base` | `aggregation.sum` | the shared weak Base | identical to `sum_scratch` |
| `neg_scratch` | `scalar.neg` | the same deterministic random model state | operands ±8, full trace |
| `neg_base` | `scalar.neg` | the shared weak Base | identical to `neg_scratch` |

The scratch outputs are audited to prove that their `shared_initial.pt` model-state hashes are identical. The Base-initialized outputs retain a parent contract containing the Base fingerprint and model-state hash. Retention KL and parameter anchoring are zero in all four conditions, so initialization is the intended difference within each operator pair.

Run unattended:

```bash
bash scripts/run_d5_operator_unit_factory.sh detach
```

Inspect status:

```bash
bash scripts/run_d5_operator_unit_factory.sh status
```

The worker resumes completed model directories, retries unexpected failures, evaluates each validation-NLL-selected checkpoint with validation seed `705100`, and writes:

```text
evaluations/d5_operator_unit_factory/summary.json
```

Interpretation is fixed:

- scratch pass, Base fail: Base initialization interferes;
- scratch fail, Base pass: Base initialization helps;
- both pass: initialization is not blocking;
- both fail: investigate task data, objective, numeric representation, or capacity.

D5 is diagnostic. It never authorizes production and never opens `iid_test`, `operand_ood`, or `length_ood`.
