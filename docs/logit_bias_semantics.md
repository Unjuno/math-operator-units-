# Logit Bias Fusion Semantics

## 1. Project scope

This repository studies **bias fusion for language models**.

The goal is to combine logit-space changes from multiple models or model variants into one next-token distribution. Mathematical operator models are used only as a controlled experimental system because their data, intermediate states, answers, and failure cases can be generated and verified exactly.

The project does not define bias fusion as routing, calibration, or correction. Those mechanisms may be tested later as alternative ways to reduce fusion error.

## 2. Central objects

Let a common base model produce logits over vocabulary `V` for prefix `x`:

```text
z_base(v | x)
```

Let specialist model `k` produce:

```text
z_k(v | x)
```

Define its bias field relative to the same base:

```text
B_k(v | x) = z_k(v | x) - z_base(v | x)
```

A fusion rule `F` combines one or more fields:

```text
B_fused(v | x) = F(B_1, B_2, ..., B_n)(v | x)
```

The fused logits and distribution are:

```text
z_fused(v | x) = z_base(v | x) + B_fused(v | x)
p_fused(v | x) = softmax(z_fused(v | x))
```

The semantic effect of fusion is the induced change in the next-token distribution:

```text
Delta p = p_fused - p_base
```

## 3. Raw fusion baseline

The first baseline is direct additive fusion:

```text
B_raw = sum_k alpha_k B_k
z_raw = z_base + B_raw
```

where `alpha_k` is fixed or searched explicitly.

Raw fusion must be measured before adding a router, gate, confidence estimator, or learned corrector. Otherwise it is impossible to determine whether composition works or whether a separate mechanism simply selected one model.

Raw addition is a baseline, not a claim of safety or correctness.

## 4. Joint reference model

A joint reference model is trained on the union of the specialist training distributions.

For the same prefix:

```text
p_joint(v | x) = softmax(z_joint(v | x))
```

A distribution-matching loss can be defined as:

```text
L_match = D(p_joint, p_fused)
```

Possible choices include KL divergence or Jensen-Shannon divergence.

```text
L_match = 0
```

means only that the two distributions are identical under the selected divergence. It does not prove that either model is correct. Exact task accuracy, intermediate-step validity, and termination behavior must be evaluated separately.

## 5. Error amplification

The main practical risk is that an irrelevant specialist does not produce a neutral field.

```text
irrelevant input
    -> structured non-zero specialist output
    -> structured bias field
    -> incorrect contribution to fusion
```

When several incorrect fields align on the same token, fusion can amplify the error:

```text
B_1(v_wrong | x) > 0
B_2(v_wrong | x) > 0
...
B_raw(v_wrong | x) becomes large
```

This is different from independent zero-mean noise. It must be measured directly.

Relevant measurements include:

- inactive bias norm;
- inactive top probability;
- incorrect-token amplification;
- field agreement and conflict;
- raw-fusion task accuracy;
- divergence from the joint reference;
- change across training checkpoints;
- value and length OOD behavior.

## 6. Why the common base matters

If each specialist is compared directly with random initialization, its field contains both:

- common language or trace-format learning;
- operator-specific learning.

Then adding several specialist fields may add the same common component repeatedly.

To isolate specialist changes, the intended construction is:

```text
random initialization
        ↓
trained common BaseGPT
        ↓
BaseGPT -> specialist 1
BaseGPT -> specialist 2
...
BaseGPT -> joint reference
```

The bias origin is then:

```text
B_k = z_k - z_base
```

where `z_base` is produced by the trained common BaseGPT checkpoint, not by an untrained random model.

## 7. Fusion rules to test

The minimal sequence of experiments is:

### 7.1 Direct sum

```text
B_fused = sum_k B_k
```

### 7.2 Mean or fixed scaling

```text
B_fused = (1 / n) sum_k B_k
```

or:

```text
B_fused = sum_k alpha_k B_k
```

with fixed reported coefficients.

### 7.3 Centered or normalized variants

```text
Center(B) = B - mean_v(B(v))
```

Normalization is an ablation because it changes field scale and may remove useful magnitude information.

### 7.4 Oracle weighting

Oracle applicability or task identity may be used only as an upper-bound control. It is not evidence that unguided bias fusion works.

### 7.5 Learned routing or calibration

Learned weights, token-wise correction, or removal fields are later hypotheses. They must be compared against the same raw baseline and the same checkpoints.

## 8. Why mathematical operators are used

Mathematical operator tasks are selected because they make the fusion experiment easier to verify:

1. training data can be generated deterministically;
2. final answers can be checked exactly;
3. intermediate equality steps can be verified;
4. mixed and conflicting operator distributions can be constructed deliberately;
5. specialist and joint training exposure can be counted exactly;
6. inactive-model behavior can be measured without subjective labels;
7. failure cases can be reproduced across checkpoints and seeds.

The operators are experimental instruments. The intended later application is more general language-model bias fusion.

## 9. What this project does not assume

The repository does not assume that:

- raw addition succeeds;
- irrelevant models are neutral;
- a corrector is always required;
- a router is equivalent to fusion;
- the joint model is necessarily optimal;
- matching the joint distribution guarantees task correctness;
- proxy-model results transfer to GPT models;
- mathematical success automatically transfers to natural language.

## 10. Minimal research claim

The current research claim is deliberately limited:

```text
Bias fusion can be studied as a controlled comparison between a common base model,
independently specialized models, and a jointly trained reference model.
Mathematical operator tasks are used because they expose useful contributions,
irrelevant leakage, interaction errors, and error amplification in a form that can
be generated and verified exactly.
```

Japanese:

```text
本プロジェクトの目的は、複数の言語モデルまたはモデル変種が作るlogit biasを融合し、
一つの次token分布を構成できるかを検証することである。数学演算子モデルは最終目的ではなく、
正解・途中状態・失敗を厳密に生成・検証できる実験系として用いる。raw fusion、誤差増幅、
joint modelとの分布差、必要に応じたroutingや補正を、同じcheckpoint上で分離して評価する。
```