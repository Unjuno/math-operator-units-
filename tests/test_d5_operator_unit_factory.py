from pathlib import Path

from opfusion.training.design_config import load_design_run_config, model_design

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs/experiments/d5_operator_unit_factory"


def load(name: str):
    return load_design_run_config(CONFIG_DIR / f"{name}.yaml")


def test_d5_configs_load_and_keep_the_comparison_controlled() -> None:
    shared = load("shared_base")
    sum_scratch = load("sum_scratch")
    sum_base = load("sum_base")
    neg_scratch = load("neg_scratch")
    neg_base = load("neg_base")

    for config in (shared, sum_scratch, sum_base, neg_scratch, neg_base):
        config.validate()
        assert config.seeds == (0,)
        assert config.require_cuda is True
        assert config.deterministic_algorithms is True
        assert model_design(config).strict_experiment_fingerprint is True
        assert model_design(config).specialist_retention_kl_weight == 0.0
        assert model_design(config).specialist_parameter_anchor_weight == 0.0

    assert shared.base_model_id == "base.common"
    assert model_design(shared).base_target_mode == "weak_multitask"
    assert sum_scratch.base_model_id is None
    assert neg_scratch.base_model_id is None
    assert sum_base.base_model_id == "base.common"
    assert neg_base.base_model_id == "base.common"

    for scratch, base in ((sum_scratch, sum_base), (neg_scratch, neg_base)):
        assert scratch.model_config == base.model_config
        assert scratch.tokenizer_config == base.tokenizer_config
        assert scratch.optimizer == base.optimizer
        assert scratch.data == base.data
        assert scratch.max_steps == base.max_steps == 3000
        assert scratch.effective_batch_size == base.effective_batch_size == 128

    assert (sum_scratch.data.operand_min, sum_scratch.data.operand_max) == (-8, 8)
    assert (sum_scratch.data.min_terms, sum_scratch.data.max_terms) == (3, 3)
    assert (neg_scratch.data.operand_min, neg_scratch.data.operand_max) == (-8, 8)
    assert sum_scratch.data.full_trace_weight == neg_scratch.data.full_trace_weight == 100
    assert sum_scratch.data.continuation_weight == neg_scratch.data.continuation_weight == 0
    assert sum_scratch.data.terminal_weight == neg_scratch.data.terminal_weight == 0
    assert sum_scratch.data.randomized_train_reduction is False
    assert neg_scratch.data.randomized_train_reduction is False


def test_d5_launcher_and_worker_exist() -> None:
    assert (ROOT / "scripts/run_d5_operator_unit_factory.sh").is_file()
    assert (ROOT / "scripts/d5_operator_unit_factory.py").is_file()
