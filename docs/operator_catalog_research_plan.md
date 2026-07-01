# Operator Catalog Research Plan

The project needs a source-backed operator backlog. The goal is not to claim that all mathematical operators can be listed. Mathematics admits infinitely many operators. The practical goal is to build a broad, curated operator catalog for model training, registry design, and tokenizer capacity planning.

## 1. Research principle

```text
Do not invent the entire catalog from memory.
Use existing operator catalogs as extraction sources.
Then normalize into this project's typed registry schema.
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
notes:
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

These should not all become learned units. Many should remain spec-only, program-only, tool-backed, or verifier-backed.

## 4. Normalization rules

### 4.1 Namespace before name

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

### 4.2 Operator kind

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

### 4.3 Implementation mode

```text
unit:
  learned primitive unit

program:
  composed from other operators

distilled_unit:
  derived operator distilled into a unit

tool:
  external tool-backed operation

spec_only:
  listed but not implemented yet
```

## 5. Priority tiers

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

## 6. Output files to create

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

## 7. Acceptance rule

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
```

## 8. Final rule

```text
Search broadly.
Normalize aggressively.
Train narrowly.
Fuse task-specific subsets.
```
