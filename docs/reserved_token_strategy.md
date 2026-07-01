# Reserved Token Strategy

Reserved operator tokens are required because tokenizer growth is a breaking change.

The goal is to allow the operator registry to grow while keeping tokenizer shape, token IDs, embedding matrices, output heads, and fusion output spaces compatible.

## 1. Core idea

A reserved token is a real token that exists from the first tokenizer release, but has no assigned operator semantics yet.

Example:

```text
<OP_RESERVED_0000>
<OP_RESERVED_0001>
...
<OP_RESERVED_4095>
```

Later, a registry version may assign one reserved token to a new operator:

```yaml
reserved_token: <OP_RESERVED_0137>
assigned_to: calc.curvature
assigned_in_registry_version: 3
```

This does not change the vocabulary shape.

## 2. What reserved tokens solve

Reserved tokens solve tokenizer and checkpoint compatibility.

They preserve:

```text
- vocabulary size
- token ID order
- embedding matrix shape
- output-head shape
- logit vector alignment
- fusion compatibility
```

This means old and new units can still have the same tensor shapes and the same output-space ABI.

## 3. What reserved tokens do not solve

Reserved tokens do not make old checkpoints understand the new operator.

If `<OP_RESERVED_0137>` was unassigned during training, old models saw it only as an unused or meaningless token.

After assigning it to `calc.curvature`, an old checkpoint is shape-compatible but not semantically trained for curvature.

Therefore:

```text
Reserved token = compatibility slot
Reserved token ≠ learned capability
```

A newly assigned operator still requires one of:

```text
- training a new unit
- training a corrector for the new unit
- distilling a program into a unit
- using program-only representation
- using fallback spelling until a unit exists
```

## 4. Compatibility levels

### Level 0: tokenizer-compatible

The checkpoint has the same tokenizer profile and vocab hash.

It can be loaded and fused at tensor level.

### Level 1: registry-compatible

The checkpoint knows the registry version and all assigned tokens referenced by its metadata.

It can resolve operator IDs and token assignments.

### Level 2: capability-compatible

The checkpoint was actually trained for the operator or for a program using that operator.

It can use the operator meaningfully.

Reserved tokens mostly guarantee Level 0 and support Level 1. They do not automatically guarantee Level 2.

## 5. Assignment rules

A reserved token may be assigned only once.

Rules:

```text
1. reserved -> assigned is allowed.
2. assigned -> different assignment is forbidden.
3. assigned -> unassigned is forbidden for released versions.
4. assignment must include operator id, registry version, date, and rationale.
5. high-priority expected operators should get direct named tokens, not only reserved tokens.
```

## 6. When to use direct tokens vs reserved tokens

Use direct named tokens for operators expected to become central:

```text
<OP_SCALAR_MUL>
<OP_SCALAR_DIV>
<OP_LINALG_MATMUL>
<OP_CALC_LAPLACIAN>
<OP_PROB_SOFTMAX>
<OP_BIAS_REMOVE>
```

Use reserved tokens for operators that are plausible but not yet stable:

```text
new geometry operators
new PDE residual families
experimental symbolic rewrite operators
new semantic/tool adapters
```

Use fallback spelling for unstable concepts:

```text
<OP_START> experimental.some_new_operator <OP_END>
```

## 7. Why not reserve only a few tokens

If the final system may mix many units, the tokenizer must anticipate growth.

The cost of too few reserved slots is high:

```text
new tokenizer version
checkpoint migration
embedding/output-head reshaping
fusion incompatibility
CI split across tokenizer universes
```

The cost of many reserved slots is mainly embedding/output-head size.

For this project, compatibility is more important than minimal vocabulary size.

## 8. Recommended policy

```text
1. Include many reserved operator tokens in tokenizer_core_v1.
2. Give direct names to expected central operators before freeze.
3. Keep rare/speculative operators as reserved or fallback.
4. Never add non-reserved tokens to tokenizer_core_v1 after release.
5. Treat reserved token assignment as registry evolution, not tokenizer evolution.
6. Train new units when reserved tokens receive real semantics.
```

## 9. Final distinction

```text
Tokenizer freeze protects compatibility.
Registry assignment gives names and semantics.
Training creates capability.
Correctors make fusion stable.
Verifiers decide correctness.
```
