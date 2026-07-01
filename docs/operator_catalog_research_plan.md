# Operator Catalog Research Plan

The project needs a source-backed operator backlog. The goal is not to claim that all mathematical operators can be listed or used. Mathematics admits infinitely many operators, and many named operations are not suitable for learned operator-unit fusion.

The practical goal is to build a broad, curated operator catalog for registry design and tokenizer capacity planning, then pass candidates through an applicability gate before training or fusion.

## 1. Research principle

```text
Do not invent the entire catalog from memory.
Use existing operator catalogs as extraction sources.
Normalize into this project's typed registry schema.
Then accept only evaluable/trainable operators into learned-unit work.
```

Every candidate operator should eventually have:

```yaml
opcode:
canonical_token:
source:
source_name:
kind:
domain:
type_signature:
implementation_mode:
priority:
applicability:
  level:
  supervision_source:
  evaluator_fn:
  verifier_fn:
  train_distribution:
  negative_distribution:
  domain_restrictions:
  failure_modes:
notes:
```

Important distinction:

```text
source catalog candidate ≠ learned unit candidate ≠ runtime fusion unit
```

## 2. Primary source catalogs

### 2.1 ONNX operator catalog

Purpose:

```text
ML graph, tensor, neural-network, quantization, reduction, shape, sequence, control-ish operators.
```

Useful groups:

```text
arithmetic: Add, Sub, Mul, Div, Pow
trig/elementary: Sin, Cos, Exp, Log, Sqrt
compare: Equal, Greater, Less
reduction: ReduceSum, ReduceMean, ReduceMax, ReduceMin
shape/tensor: Reshape, Transpose, Slice, Gather, Scatter, Concat, Split
linalg-ish: MatMul, Det, Einsum
prob/NN: Softmax, LogSoftmax, Dropout, BatchNormalization, Attention
quantization: QuantizeLinear, DequantizeLinear
```

Action:

```text
Extract ONNX operators into configs/operators/source_catalogs/onnx.yaml.
Map only relevant operators into math_expansion_backlog.yaml or tensor_expansion_backlog.yaml.
Do not mark an operator trainable unless an evaluator, verifier, program, or dataset scoring rule exists.
```

### 2.2 MLIR dialects

Purpose:

```text
Compiler/operator namespace design and typed IR structure.
```

Useful dialect families:

```text
arith
math
complex
linalg
tensor
vector
shape
scf
cf
affine
func
memref
sparse_tensor
```

Action:

```text
Use MLIR as a namespace sanity check: arithmetic, tensor, control-flow, shape, memory, vector, and linalg should be distinct.
```

### 2.3 NumPy ufuncs

Purpose:

```text
Elementwise numeric operators, comparison operators, bitwise operators, floating-point classification, trigonometric and special elementary functions.
```

Useful groups:

```text
math ufuncs
trigonometric ufuncs
bit-twiddling functions
comparison functions
floating functions
```

Action:

```text
Extract elementwise scalar/vector operators and normalize into scalar or tensor namespaces.
Treat them as trainable only when synthetic data and exact/numeric evaluators are available.
```

### 2.4 SymPy functions and modules

Purpose:

```text
Symbolic math, special functions, combinatorics, discrete math, calculus, solvers, matrices, geometry, logic.
```

Useful groups:

```text
elementary functions
combinatorial functions
special functions
calculus operations
simplification / expansion / factorization
polynomial operations
matrix symbolic operations
logic
sets
geometry
```

Action:

```text
Use SymPy to extend symbolic, special-function, combinatorics, and geometry operator candidates.
Prefer program-only, verifier-backed, or spec-only modes unless exact generation or symbolic verification is available.
```

## 3. Secondary source catalogs

These should be searched next.

```text
SciPy special functions
SciPy sparse / linalg / optimize / signal operators
PyTorch torch.ops / ATen operators
JAX lax operators
TensorFlow raw ops
NetworkX algorithms
SMT-LIB theories
Lean / Coq / Isabelle primitive constructs
Wolfram built-in function categories
OpenQASM / quantum gate sets
GraphBLAS operators
SQL relational algebra operators
```

These should not all become learned units. Many should remain spec-only, program-only, tool-backed, dataset-backed, or verifier-backed.

## 4. Applicability gate

The catalog pipeline has five levels.

```text
L0 candidate:
  name collected from a source catalog

L1 specified:
  canonical token, kind, domain, and type signature exist

L2 evaluable:
  exact evaluator, numeric evaluator, verifier, program definition, or dataset scoring rule exists

L3 trainable:
  train/validation/test distributions and negative distributions are defined

L4 fusible:
  unit + corrector + metrics exist and the unit is allowed in runtime fusion manifests
```

Rules:

```text
- L0/L1 entries may be kept for planning.
- L2 entries may be used for evaluation design.
- L3 entries may be trained.
- L4 entries may be fused.
```

Final gate:

```text
No rule, no evaluator, no verifier, no dataset regularity -> no learned unit.
```

## 5. Applicability sources

An operator may advance only if one of these supervision sources exists.

```text
exact_generator:
  exact programmatic target generation

numeric_evaluator:
  numeric target generation with tolerance and domain restrictions

verifier:
  output can be checked even if generation is hard

dataset:
  stable pattern can be learned from curated data and scoring rules

program:
  operator is a composition over existing operators

tool:
  external tool-backed operation with reproducible interface

none:
  may remain candidate/spec-only but cannot become learned unit
```

## 6. Normalization rules

### 6.1 Namespace before name

Ambiguous names must be split.

```text
Div:
  OP_SCALAR_DIV
  OP_CALC_DIVERGENCE

Max:
  OP_SCALAR_MAX2
  OP_AGG_MAX
  OP_BIAS_MAX

Proj:
  OP_LINALG_PROJ_VECTOR
  OP_OPT_PROJECT_CONSTRAINT
  OP_BIAS_PROJ
```

### 6.2 Operator kind

Use these categories unless there is a strong reason to add a new one.

```text
math_exact
math_numeric
math_discrete
math_symbolic
math_functional
math_tensor
math_graph
math_geometric
bias_algebra
fusion_control
verifier
candidate_search
non_math_semantic
non_math_tool
```

### 6.3 Implementation mode

```text
unit:
  learned primitive unit, allowed only at L3+

program:
  composed from other operators

distilled_unit:
  derived operator distilled into a unit, allowed only at L3+

tool:
  external tool-backed operation

spec_only:
  listed but not implemented yet

dataset_backed:
  learned from curated data and scoring rules; not math_exact
```

## 7. Priority tiers

```text
S0:
  already core / immediate experiments

S1:
  needed for first serious composition benchmarks

S2:
  needed for broad numeric/math benchmark

S3:
  useful but harder/discontinuous/specialized

S4:
  symbolic/scientific/solver-heavy

S5:
  tool, semantic, external-state, or speculative
```

Priority does not override applicability. A high-priority operator still cannot become a learned unit without L2/L3 evidence.

## 8. Output files to create

```text
configs/operators/source_catalogs/onnx.yaml
configs/operators/source_catalogs/mlir.yaml
configs/operators/source_catalogs/numpy_ufuncs.yaml
configs/operators/source_catalogs/sympy.yaml
configs/operators/source_catalogs/scipy.yaml
configs/operators/source_catalogs/pytorch_aten.yaml
configs/operators/source_catalogs/jax_lax.yaml
configs/operators/source_catalogs/networkx.yaml
configs/operators/source_catalogs/smtlib.yaml
```

Then merge curated entries into:

```text
configs/operators/math_expansion_backlog.yaml
configs/operators/tensor_expansion_backlog.yaml
configs/operators/symbolic_expansion_backlog.yaml
configs/operators/graph_expansion_backlog.yaml
configs/operators/custom_operator_backlog.yaml
```

## 9. Acceptance rule

A source-catalog entry is accepted into the main backlog only if it has:

```text
- clear name
- unambiguous canonical token
- type signature
- source reference
- domain
- kind
- implementation mode
- priority
- applicability level
- supervision source
```

It is accepted into learned-unit training only if it also has:

```text
- evaluator, verifier, program definition, or dataset scoring rule
- train distribution
- validation/test distribution
- negative distribution
- domain restrictions
- known failure modes
```

It is accepted into runtime fusion only if it has:

```text
- trained unit
- trained corrector or explicit no-corrector justification
- required metrics
- tokenizer/profile compatibility
- registry assignment compatibility
```

## 10. Final rule

```text
Search broadly.
Specify carefully.
Evaluate strictly.
Train narrowly.
Fuse only validated units.
```
