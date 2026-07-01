# Unassigned Reserved Token Policy

Reserved tokens exist to preserve tokenizer and checkpoint compatibility, but an unassigned reserved token must be inert.

## 1. Core rule

```text
Unassigned reserved tokens must not create active model behavior.
```

They are compatibility slots, not learned capabilities.

## 2. Input behavior

If an unassigned reserved operator token appears in model input, the system should treat it as an invalid or unknown operator.

Required behavior:

```text
- all operator-unit gates should close
- no operator-specific bias should be preserved
- the system may abstain or route to fallback spelling
- the event should be counted as invalid/reserved-token activation
```

In fusion terms:

```text
g_k(x) ≈ 0 for all ordinary operator units
```

## 3. Output behavior

If unassigned reserved tokens exist in the output vocabulary, they must not be allowed to receive probability mass during normal decoding or classification.

Required behavior:

```text
- mask unassigned reserved output tokens
- or set their logits to a large negative value
- or exclude them from task-specific candidate sets
```

The preferred policy is candidate masking:

```text
z[token] = -inf for every unassigned reserved output token
```

This prevents random output-head values from leaking into the final distribution.

## 4. Bias behavior

For a unit that is not trained to use an unassigned reserved token, the contribution for that token should be neutral.

For bias/logit fusion:

```text
b_k[unassigned_reserved_token] = 0
```

or, if logits are centered over a task-specific candidate set, the reserved token should be outside the candidate set entirely.

## 5. Random initialization is not a policy

A random untrained output head may assign arbitrary logits to reserved tokens.

This is not acceptable as runtime behavior.

Therefore:

```text
random reserved-token logits must be masked or neutralized
```

The system must not rely on random initialization being harmless.

## 6. When a reserved token becomes assigned

When a reserved token is assigned to a real operator, it changes from inert to meaningful at the registry level.

However, capability still requires training.

```text
reserved token assignment -> semantics exist in registry
unit training -> capability exists in model
corrector training -> stable fusion behavior exists
```

Old checkpoints remain shape-compatible but are not automatically capability-compatible.

## 7. Metrics

The following metrics should be added for reserved-token safety:

```text
reserved_input_gate_mean
reserved_input_inactive_leakage
reserved_output_probability_mass
reserved_output_pmax
reserved_mask_coverage
```

Expected values for unassigned reserved tokens:

```text
reserved_input_gate_mean ≈ 0
reserved_input_inactive_leakage ≈ 0
reserved_output_probability_mass ≈ 0
reserved_output_pmax ≈ 0
reserved_mask_coverage = 1
```

## 8. CI requirements

CI must test:

```text
1. unassigned reserved tokens are present in vocab
2. unassigned reserved tokens are marked unassigned in registry
3. unassigned reserved tokens are masked from outputs
4. unassigned reserved tokens do not activate ordinary operator units
5. assigned reserved tokens are never reassigned
```

## 9. Final statement

```text
Reserved tokens preserve shape.
Unassigned reserved tokens are inert.
Assigned reserved tokens need training.
Fusion only uses active, assigned, trained semantics.
```
