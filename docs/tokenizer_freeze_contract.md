# Tokenizer Freeze Contract

This document defines the non-negotiable compatibility contract for tokenizers in this repository.

The short version is:

```text
If units will ever be fused, their tokenizer profile and vocab hash must match exactly.
```

## 1. Why this contract exists

The project eventually aims to mix many operator units:

```text
z_final = z_0 + Σ_k g_k(x) b_k(x)
```

This is only valid when every unit's output vector is defined over the same token IDs in the same order.

If token ID 120 means `<OP_SCALAR_ADD>` for one unit but `<OP_BIAS_CENTER>` for another unit, fusion becomes mathematically invalid even if tensor shapes match.

Therefore, tokenizer compatibility is not a convenience rule. It is part of the mathematical definition of the fused system.

## 2. Hard invariants

The following invariants must never be violated within a tokenizer major version.

```text
1. Token IDs are immutable.
2. Token strings are immutable.
3. Token IDs are never reused.
4. Token meanings are never reassigned.
5. Vocabulary order is immutable.
6. Reserved slots are real tokens from day one.
7. Assigned reserved slots are never reassigned.
8. Checkpoints must record tokenizer profile and vocab hash.
9. Fusion must reject mismatched vocab hashes.
10. Adding a non-reserved token requires a tokenizer major version bump.
```

## 3. Forbidden changes

These changes are forbidden in `tokenizer_core_v1`:

```text
- inserting a token into the middle of the vocabulary
- changing an existing token string
- changing an existing token ID
- deleting a token
- reusing an old token ID for a new operator
- changing a token's semantic assignment
- changing reserved slot count
- changing numeric token range
- changing register token range
- changing type-token names
- changing byte/fallback token IDs
```

Any of these requires `tokenizer_core_v2`.

## 4. Allowed changes

The following changes are allowed without changing tokenizer shape:

```text
- adding a new registry operator as program-only
- assigning an unassigned reserved token to a new primitive operator
- adding aliases in registry metadata
- adding documentation
- adding evaluator functions
- adding model checkpoints that use the same vocab hash
- adding non-token metadata to registry entries
```

Reserved-token assignment is allowed because the token already exists in the vocabulary.

## 5. Reserved-token assignment rule

A reserved operator token may move from `unassigned` to `assigned`, but never back or sideways.

Example:

```yaml
reserved_token: <OP_RESERVED_0137>
status: assigned
assigned_to: calc.curvature
assigned_in_registry_version: 3
```

After this assignment, `<OP_RESERVED_0137>` can never be assigned to another operator.

## 6. Tokenizer profiles

The repository may define multiple tokenizer profiles, but they are separate compatibility universes.

```text
tokenizer_toy_v0:
  small smoke-test tokenizer
  not compatible with main fusion artifacts

tokenizer_core_v1:
  default tokenizer for main scalar, bias, and control units

tokenizer_full_v1:
  larger tokenizer for long-term math/non-math operator expansion
```

A checkpoint must declare:

```json
{
  "tokenizer_profile": "tokenizer_core_v1",
  "tokenizer_version": "1.0.0",
  "vocab_hash": "sha256:..."
}
```

## 7. Model-size implication

A large fixed vocabulary increases embedding and output-head size.

For tiny proxy experiments, it is acceptable to use tensor inputs or restricted output heads, but the checkpoint must then be marked as profile-specific and not silently fused with full-vocab units.

Recommended separation:

```text
operator/control tokens:
  fixed symbolic tokenizer

numeric values:
  atomic integer tokens for toy experiments
  digit tokens for arithmetic generalization
  tensor side-channel for continuous/vector/PDE experiments

large semantic text:
  byte/fallback tokens or external text tokenizer
```

Do not force all continuous mathematics into text tokens. The symbolic operator interface should be fixed; tensors may carry values.

## 8. Fusion loader requirement

Fusion code must check:

```python
assert unit_a.tokenizer_profile == unit_b.tokenizer_profile
assert unit_a.vocab_hash == unit_b.vocab_hash
assert unit_a.output_space_id == unit_b.output_space_id
```

If any assertion fails, fusion must fail closed.

## 9. CI requirements

CI must enforce:

```text
- tokenizer files are immutable unless tokenizer major version changes
- no duplicate token strings
- no duplicate token IDs
- registry tokens exist in tokenizer vocab
- reserved assignments are one-to-one
- program-only operators do not require direct tokens
- checkpoints include tokenizer profile and vocab hash
- fusion rejects mismatched vocab hash fixtures
```

## 10. Final rule

```text
Tokenizer first. Registry second. Models third.
```

The tokenizer defines the ABI.
The registry defines the operator semantics.
The checkpoints implement registry entries under a specific tokenizer ABI.
