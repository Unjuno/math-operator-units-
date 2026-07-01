# Tokenizer Design

## 1. Principle

This repository uses a fixed tokenizer for operator-unit experiments.

The tokenizer is not only text preprocessing. It defines the shared symbolic interface between:

- operator registry
- data generators
- model input embeddings
- output logits
- corrector / gate models
- fusion modules
- saved checkpoints
- reproducibility tests

Therefore, tokenizer changes are treated as **breaking changes**.

Core rule:

```text
Registry can grow.
Tokenizer v1 must not grow.
```

New operators must use one of the following mechanisms:

1. pre-existing direct operator tokens,
2. pre-reserved operator slots,
3. program definitions over existing primitive operators,
4. fallback spelling tokens,
5. or a new tokenizer major version.

## 2. Why token IDs are part of the ABI

For GPT-like or token-classification models, vocabulary size affects model shape.

If the vocabulary size is `|V|`, the embedding matrix is:

```text
E ∈ R^{|V| × d}
```

and the output head is usually:

```text
W_out ∈ R^{d × |V|}
```

Adding tokens changes `|V|`, which changes checkpoint tensor shapes and logit alignment.

Always-on fusion requires every unit to emit compatible bias/logit vectors:

```text
z_final = z_0 + Σ_k g_k(x) b_k(x)
```

This only works if all `b_k(x)` are defined over the same fixed output space.

Therefore:

```text
Token IDs are part of the model ABI.
```

## 3. Versioning policy

```text
tokenizer_core_v1:
  fixed token IDs
  fixed special tokens
  fixed operator namespaces
  fixed reserved slots

tokenizer_core_v2:
  breaking change
  explicit migration required
```

Rules:

```text
1. Token strings must never be reassigned to different IDs.
2. Token IDs must never be reused for different meanings.
3. New operator names must not be inserted into the middle of the vocabulary.
4. If a new token is necessary and no reserved slot applies, create tokenizer_v2.
5. tokenizer_v2 checkpoints must be stored separately from tokenizer_v1 checkpoints.
```

## 4. Registry vs tokenizer

The operator registry may contain many operators, but not every registry operator needs a unique model-facing token.

Representation modes:

| mode | meaning | token requirement |
|---|---|---|
| `primitive_token` | direct primitive operator token | fixed token required |
| `program` | defined by other operators | no direct token required |
| `reserved_assigned` | uses a preallocated reserved token | no tokenizer shape change |
| `fallback_spelling` | spelled with fallback tokens | no tokenizer shape change |

Example derived operator:

```yaml
opcode: "<SUB>"
implementation_mode: "program"
program:
  - "x1 = <OP_SCALAR_NEG>(rhs)"
  - "out = <OP_SCALAR_ADD>(lhs, x1)"
```

## 5. Token classes

### 5.1 Special tokens

```text
<PAD>
<BOS>
<EOS>
<UNK>
<MASK>
<SEP>
<OUT>
<ERR>
<ABSTAIN>
<NOOP>
```

### 5.2 Structural tokens

```text
<LPAREN>
<RPAREN>
<LBRACK>
<RBRACK>
<LBRACE>
<RBRACE>
<COMMA>
<COLON>
<SEMICOLON>
<ASSIGN>
<ARROW>
<PIPE>
```

### 5.3 Register / variable tokens

```text
<X0> ... <X255>
<A>
<B>
<C>
<D>
<U>
<V>
<W>
<OUT_VAR>
```

Registers are preferred over arbitrary variable names because they keep parsing fixed and avoid tokenizer growth.

### 5.4 Type tokens

```text
<T_SCALAR>
<T_INT>
<T_REAL>
<T_BOOL>
<T_VECTOR>
<T_MATRIX>
<T_TENSOR>
<T_SET>
<T_FUNCTION>
<T_FIELD>
<T_DISTRIBUTION>
<T_BIAS>
<T_TEXT>
<T_DOCS>
<T_TOOL_RESULT>
```

Type tokens are important because invalid operator activation is a major failure mode.

The corrector gate should learn a type-compatible suppression rule:

```text
g_k(x) = g_pattern,k(x) · g_type,k(x)
```

If the type is incompatible, the unit should be suppressed.

### 5.5 Numeric tokens

For early experiments, use fixed integer tokens:

```text
<N_-512> ... <N_0> ... <N_512>
```

Optional extensions:

```text
<SIGN_PLUS>
<SIGN_MINUS>
<DIGIT_0> ... <DIGIT_9>
<DECIMAL>
<FRAC_BAR>
<EXP_MARK>
```

Recommended policy:

```text
scalar toy experiments:
  use atomic integer tokens

larger arithmetic experiments:
  add digit-level representation separately

continuous/vector experiments:
  use tensor inputs, not textual number tokens
```

### 5.6 Operator tokens

Use stable namespace prefixes:

```text
<OP_SCALAR_*>
<OP_COMPARE_*>
<OP_AGG_*>
<OP_LOGIC_*>
<OP_SET_*>
<OP_NUMTHEORY_*>
<OP_LINALG_*>
<OP_CALC_*>
<OP_PROB_*>
<OP_BIAS_*>
<OP_OPT_*>
<OP_PDE_*>
<OP_CTRL_*>
<OP_SEM_*>
<OP_TOOL_*>
```

Human-readable aliases such as `<ADD>` may exist in the registry, but model-facing tokens should be canonical and namespace-safe.

## 6. Reserved operator slots

Reserve unused operator tokens from the beginning:

```text
<OP_RESERVED_0000>
<OP_RESERVED_0001>
...
<OP_RESERVED_4095>
```

When a new operator is accepted into the registry, it may be assigned to a reserved token.

Rules:

```text
1. Reserved token IDs are fixed from tokenizer_v1.
2. Reserved tokens may be assigned later without changing tokenizer shape.
3. Once assigned, a reserved token must never be reassigned.
4. Assignment must be recorded in registry history.
```

## 7. Initial direct operator tokens

The initial tokenizer should include direct tokens for `core_v0`:

```text
<OP_SCALAR_ZERO>
<OP_SCALAR_ONE>
<OP_SCALAR_CONST>
<OP_SCALAR_ID>
<OP_SCALAR_NEG>
<OP_SCALAR_ADD>
<OP_SCALAR_ABS>
<OP_SCALAR_POS>
<OP_SCALAR_MIN>
<OP_SCALAR_MAX>

<OP_BIAS_ADD>
<OP_BIAS_SUB>
<OP_BIAS_CENTER>
<OP_BIAS_SCALE>
<OP_BIAS_POS>

<OP_CTRL_GATE>
<OP_CTRL_SUPPRESS>
<OP_CTRL_ABSTAIN>
```

## 8. Tokenizer profiles

Use explicit profiles. Never silently mix them.

```text
tokenizer_toy_v0:
  minimal arithmetic only
  smoke tests only
  incompatible with full registry

tokenizer_core_v1:
  recommended default
  fixed for main experiments

tokenizer_full_v1:
  includes all reserved namespaces
  intended for long-term checkpoints
```

Compatibility rule:

```text
A unit checkpoint must declare its tokenizer profile.
Fusion is allowed only between units with the same tokenizer profile and vocab hash.
```

## 9. Vocab hash

Every checkpoint must store a tokenizer hash:

```json
{
  "tokenizer_name": "ouf_core_tokenizer",
  "tokenizer_version": "1.0.0",
  "vocab_hash": "sha256:..."
}
```

Fusion loaders must reject incompatible vocab hashes:

```text
if unit_a.vocab_hash != unit_b.vocab_hash:
    reject fusion
```

## 10. CI requirements

CI must check:

```text
1. No duplicate token strings.
2. No duplicate token IDs.
3. Registry operator tokens exist in vocab.
4. Program-only operators do not require direct tokens.
5. Reserved token assignment is unique.
6. Checkpoints declare tokenizer hash.
7. Fusion rejects mismatched tokenizer hashes.
8. Tokenizer files do not change except through tokenizer version bump.
```

## 11. Final policy

```text
Tokenizer is fixed.
Operator registry can grow.
Primitive core operators get direct tokens.
Derived operators are programs.
Future operators use reserved slots.
Rare/unstable operators use fallback spelling.
Breaking changes require tokenizer_v2.
Fusion requires identical vocab hash.
```
