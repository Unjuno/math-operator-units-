from pathlib import Path

from opfusion.audit import audit_repo
from opfusion.tokenizer import FixedVocabTokenizer
from opfusion.training.config import load_run_config
from opfusion.training.data import EXPERIMENT_OPERATORS, SyntheticTraceFactory


ROOT = Path(__file__).parents[1]
PRIMARY = ROOT / "configs/experiments/gpt_bias_fusion_factory_surface_v3.yaml"


def test_repository_audit_passes_for_the_canonical_surface_experiment() -> None:
    report = audit_repo(ROOT, data_samples_per_operator=16)
    assert report["status"] == "passed", report["errors"]
    assert all(report["shared_prefix_checks"].values())


def test_common_base_and_specialists_use_identical_model_facing_prefixes() -> None:
    run = load_run_config(PRIMARY)
    tokenizer = FixedVocabTokenizer.from_config(ROOT / run.tokenizer_config)
    factory = SyntheticTraceFactory(tokenizer, run.data)
    for index, operator_id in enumerate(EXPERIMENT_OPERATORS):
        kwargs = dict(
            seed=123,
            split="validation",
            step=index,
            sample_index=index,
            forced_operator=operator_id,
        )
        base = factory.training_example("base.common", **kwargs)
        specialist = factory.training_example(operator_id, **kwargs)
        assert base.prompt_tokens == specialist.prompt_tokens
        assert base.task == "identity_equivalence"
        assert base.final_value is None
        assert specialist.final_value is not None


def test_typed_v2_launcher_requires_explicit_opt_in() -> None:
    launcher = (ROOT / "scripts/run_bias_fusion_factory_v2.sh").read_text(encoding="utf-8")
    assert "OPFUSION_ALLOW_TYPED_V2" in launcher
    assert "run_bias_fusion_factory_surface_v3.sh" in launcher


def test_arch_bootstrap_advertises_only_the_canonical_production_command() -> None:
    bootstrap = (ROOT / "scripts/bootstrap_arch_linux.sh").read_text(encoding="utf-8")
    assert "run_bias_fusion_factory_surface_v3.sh" in bootstrap
    assert "gpt_bias_fusion_factory_surface_v3.yaml" in bootstrap
    assert "run_bias_fusion_factory_v2.sh" not in bootstrap
