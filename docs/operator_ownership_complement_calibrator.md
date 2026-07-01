# Operator Ownership Complement Calibrator

This note defines a simple learned calibrator based on operator ownership.

## 1. Core idea

For each generator `M_k`, train a paired calibrator `R_k` to answer:

```text
Does the generated path or bias field belong to M_k's operator family?
```

This is an ownership classifier:

```text
s_k = R_k(x, B_k) = P(owned_by_k | x, B_k)
```

where:

```text
B_k = M_k(x)
```

The complement is the non-owned or unreliable score:

```text
e_k = 1 - s_k
```

This should be treated as a complement, not as a reciprocal. A true reciprocal such as `1 / s_k` is unstable because it explodes when `s_k` is small.

## 2. Using the complement for attenuation

The simplest calibrated field is:

```text
B_tilde_k = s_k B_k
```

Equivalently, the removed part is:

```text
E_k = (1 - s_k) B_k
B_tilde_k = B_k - E_k
```

Thus:

```text
owned score high:
  keep the generator's field

owned score low:
  remove or attenuate the generator's field
```

## 3. Token-wise version

A stronger form predicts ownership per token direction:

```text
s_k(v | x) = P(owned_by_k at token v | x, B_k)
```

Then:

```text
B_tilde_k(v | x) = s_k(v | x) B_k(v | x)
```

and:

```text
E_k(v | x) = (1 - s_k(v | x)) B_k(v | x)
```

This allows the calibrator to preserve valid parts of the field while removing only non-owned or unreliable token directions.

## 4. Active anti-bias option

If simple attenuation is insufficient, the complement can drive an anti-bias term:

```text
B_tilde_k = s_k B_k - mu (1 - s_k) P_bad(B_k)
```

where `P_bad` projects or masks the estimated harmful component, and `mu` controls suppression strength.

This should be used carefully. The first baseline should be attenuation, not active anti-bias.

## 5. Training target

Positive examples:

```text
M_k on owned operator-family inputs
valid paths produced by M_k
fields whose softmax effect matches the reference field
```

Negative examples:

```text
M_k on other operator-family inputs
M_k on OOD inputs
M_k outputs that are high-confidence but wrong
operator assimilation errors
```

Binary target:

```text
y_owned = 1 for owned/reliable path
y_owned = 0 for non-owned/unreliable path
```

Loss:

```text
L_owned = BCE(s_k, y_owned)
```

Optionally combine with effect-space loss:

```text
L_effect = JSD(softmax(z_0 + B_tilde_k), softmax(z_0 + B_ref))
```

## 6. Runtime composition

Each generator-calibrator pair emits a calibrated field:

```text
B_tilde_1, B_tilde_2, ..., B_tilde_n
```

Then all calibrated fields are composed:

```text
F = O(B_tilde_1, B_tilde_2, ..., B_tilde_n)
```

and decoded:

```text
p_final = softmax(z_0 + lambda F)
```

## 7. Short framing

```text
Train the calibrator to predict whether the generator's emitted path belongs to that generator's operator family. Use the ownership probability to keep the field and its complement to attenuate or remove the non-owned component.
```

Japanese:

```text
補正器には、その生成モデルが出した経路やbias fieldが、そのモデル自身のoperator familyに属するかを学習させる。所属確率を保持量として使い、その補数を減衰量または除去量として使う。ここで使うのは逆数ではなく補数である。
```
