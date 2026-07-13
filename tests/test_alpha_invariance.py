from __future__ import annotations

import pytest
import torch

from opfusion.model import GPTConfig, GPTModel


def _model_and_base(config: GPTConfig | None = None) -> tuple[GPTModel, torch.Tensor, torch.Tensor]:
    cfg = config or GPTConfig()
    model = GPTModel(cfg)
    model.eval()
    input_ids = torch.randint(0, min(cfg.vocab_size, 100), (1, 32))
    base = torch.randn(1, 32, cfg.vocab_size)
    return model, input_ids, base


def test_gpt_model_param_count_within_1m() -> None:
    cfg = GPTConfig()
    model = GPTModel(cfg)
    n = model.param_count
    assert n < 1_000_000, f"param count {n} exceeds 1M"
    assert n > 100_000, f"param count {n} too small, likely broken"


def test_gpt_model_forward_shape() -> None:
    cfg = GPTConfig()
    model = GPTModel(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 64))
    logits = model(input_ids)
    assert logits.shape == (2, 64, cfg.vocab_size)


def test_gpt_model_output_finite() -> None:
    cfg = GPTConfig()
    model = GPTModel(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    logits = model(input_ids)
    assert torch.isfinite(logits).all()


def test_gate_zero_logits_unchanged() -> None:
    model, input_ids, base = _model_and_base()
    bias = model.get_bias_field(input_ids, base)
    fused = base + 0.0 * bias
    assert torch.equal(fused, base), "gate=0 must preserve base logits"


def test_gate_one_applies_full_bias() -> None:
    model, input_ids, base = _model_and_base()
    bias = model.get_bias_field(input_ids, base)
    fused = base + 1.0 * bias
    expected = base + bias
    assert torch.allclose(fused, expected), "gate=1 must apply full bias"


def test_alpha_scaling_is_linear() -> None:
    model, input_ids, base = _model_and_base()
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    bias = model.get_bias_field(input_ids, base)
    results = [base + a * bias for a in alphas]
    for a, r in zip(alphas, results):
        expected = base + a * bias
        assert torch.allclose(r, expected), f"alpha={a} should scale linearly"


def test_alpha_independent_across_units() -> None:
    cfg = GPTConfig()
    model_a = GPTModel(cfg)
    model_b = GPTModel(cfg)
    model_a.eval()
    model_b.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    base = torch.randn(1, 16, cfg.vocab_size)

    bias_a = model_a.get_bias_field(input_ids, base)
    bias_b = model_b.get_bias_field(input_ids, base)

    fused_a = base + 1.0 * bias_a
    fused_b = base + 1.0 * bias_b

    assert not torch.allclose(fused_a, fused_b), "independent units must produce different fields"


def test_weight_tying_embedding_head_shared() -> None:
    cfg = GPTConfig(weight_tying=True)
    model = GPTModel(cfg)
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_untied_separate() -> None:
    cfg = GPTConfig(weight_tying=False)
    model = GPTModel(cfg)
    assert model.lm_head.weight is not model.token_embedding.weight


def test_config_param_count_estimate() -> None:
    cfg = GPTConfig()
    model = GPTModel(cfg)
    estimate = cfg.param_count_estimate
    actual = model.param_count
    ratio = estimate / actual
    assert 0.8 < ratio < 1.2, f"estimate {estimate} vs actual {actual} ratio {ratio:.3f} off"


def test_model_generate_shape() -> None:
    cfg = GPTConfig()
    model = GPTModel(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(input_ids, max_new_tokens=4, temperature=1.0)
    assert out.shape == (1, 12)
