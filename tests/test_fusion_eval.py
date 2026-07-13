import torch

from opfusion.fusion_eval import Jensen_shannon_divergence, center_logit_field, fuse_logits


def test_empty_subset_is_exactly_the_base() -> None:
    base = torch.randn(2, 3, 7)
    fused = fuse_logits(base, [], mode="raw_sum")
    assert torch.equal(fused, base)


def test_singleton_raw_sum_is_exactly_the_specialist() -> None:
    base = torch.randn(2, 3, 7)
    specialist = torch.randn(2, 3, 7)
    fused = fuse_logits(base, [specialist], mode="raw_sum")
    assert torch.allclose(fused, specialist)


def test_raw_sum_and_bias_mean_follow_the_declared_formulas() -> None:
    base = torch.randn(2, 5)
    left = torch.randn(2, 5)
    right = torch.randn(2, 5)
    raw = fuse_logits(base, [left, right], mode="raw_sum")
    mean = fuse_logits(base, [left, right], mode="bias_mean")
    assert torch.allclose(raw, base + (left - base) + (right - base))
    assert torch.allclose(mean, base + 0.5 * ((left - base) + (right - base)))


def test_centering_does_not_change_softmax() -> None:
    field = torch.randn(4, 11)
    centered = center_logit_field(field)
    assert torch.allclose(torch.softmax(field, dim=-1), torch.softmax(centered, dim=-1), atol=1e-6)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(4), atol=1e-6)


def test_jsd_is_zero_for_identical_logits() -> None:
    logits = torch.randn(3, 4, 9)
    value = Jensen_shannon_divergence(logits, logits)
    assert float(value) >= -1e-7
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-7)
