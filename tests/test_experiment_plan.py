from pathlib import Path

import yaml

from opfusion.training.design_config import load_design_run_config, model_design

ROOT = Path(__file__).parents[1]
PLAN_V1 = ROOT / "configs/experiments/experiment_plan_v1.yaml"
PLAN_V2 = ROOT / "configs/experiments/experiment_plan_v2.yaml"


def load_plan(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["plan"]


def test_experiment_plan_v2_matches_configs() -> None:
    v1 = load_plan(PLAN_V1)
    plan = load_plan(PLAN_V2)
    assert plan["id"] == "bias_fusion_surface_v4_v2"
    assert plan["supersedes"] == v1["id"]
    assert plan["status"] == "prospective_before_target_gpu_pilot"
    assert (ROOT / plan["document"]).is_file()

    smoke = load_design_run_config(ROOT / plan["smoke"]["config"])
    assert smoke.require_cuda and smoke.seeds == (0,)
    assert 0 < smoke.max_steps <= 3

    pilots = [load_design_run_config(ROOT / path) for path in plan["pilot"]["configs"]]
    assert len(pilots) == 4
    assert all(run.seeds == (0,) for run in pilots)
    assert all(run.require_cuda for run in pilots)
    assert all(run.deterministic_algorithms and not run.allow_tf32 for run in pilots)
    assert len({run.max_steps for run in pilots}) == 1
    assert plan["pilot"]["splits"] == ["validation"]
    assert plan["pilot"]["eligibility"] == v1["pilot"]["eligibility"]
    for key in (
        "clear_winner_gap",
        "near_tie_tolerance",
        "near_tie_inactive_drift_reduction",
        "retention_minimum_drift_reduction",
        "retention_maximum_relevant_regression",
    ):
        assert plan["pilot"][key] == v1["pilot"][key]

    production = load_design_run_config(ROOT / plan["production"]["config"])
    assert list(production.seeds) == plan["production"]["seeds"] == [0, 1, 2]
    assert plan["production"]["models_per_seed"] == 7

    final = plan["final"]
    assert final["evaluation_seed"] == 700000
    assert final["splits"] == ["iid_test", "operand_ood", "length_ood"]
    assert final["examples_per_operator"] == 64
    assert final["primary_subset"] == 31
    assert final["primary_condition"] == "raw_sum"
    assert final["primary_alpha"] == 1.0
    for key in (
        "preserved_mean_gap",
        "preserved_worst_operator_gap",
        "material_failure_mean_gap",
        "material_failure_eos_gap",
    ):
        assert final[key] == v1["final"][key]
    assert final["selected_rescue_is_secondary"] is True
    assert plan["reserved_until_frozen"] == final["splits"]

    reporting = plan["reporting"]
    assert reporting["replication_unit"] == "training_seed"
    assert reporting["formal_p_values"] is False
    assert reporting["report_every_seed"] is True
    assert reporting["report_every_operator"] is True
    assert reporting["aggregate_seed_weighting"] == "equal"
    assert reporting["missing_seed_policy"] == "no_imputation"


D4_CONFIGS = [
    ROOT / "configs/experiments/d4_specialist_ablation/sum_a.yaml",
    ROOT / "configs/experiments/d4_specialist_ablation/sum_b.yaml",
    ROOT / "configs/experiments/d4_specialist_ablation/sum_c.yaml",
    ROOT / "configs/experiments/d4_specialist_ablation/neg_a.yaml",
    ROOT / "configs/experiments/d4_specialist_ablation/neg_b.yaml",
    ROOT / "configs/experiments/d4_specialist_ablation/neg_c.yaml",
]


def test_d4_specialist_ablation_configs_load() -> None:
    for path in D4_CONFIGS:
        config = load_design_run_config(path)
        assert config.experiment_id.startswith("d4_specialist_ablation_")
        assert len(config.operators) == 5
        assert tuple(config.operators) == (
            "scalar.add", "aggregation.sum", "scalar.neg", "scalar.min", "scalar.max",
        )
        assert len(config.joint_model_ids) >= 1
        assert config.seeds == (0,)
        assert config.deterministic_algorithms is True


def test_d4_specialist_ablation_conditions() -> None:
    """Verify A/B/C variant parameters match expectations."""
    for path in D4_CONFIGS:
        config = load_design_run_config(path)
        name = path.stem
        design = model_design(config)
        assert design.selection_metric == "validation_nll"

        # B and C variants: canonical left-fold (non-randomized reduction)
        if name in ("sum_b", "sum_c", "neg_b", "neg_c"):
            assert config.data.randomized_train_reduction is False, f"{name} should be canonical"
            assert config.data.full_trace_weight == 100
            assert config.data.continuation_weight == 0
            assert config.data.terminal_weight == 0
        # A variants: randomized reduction
        else:
            assert config.data.randomized_train_reduction is True, f"{name} should be randomized"
            assert config.data.full_trace_weight == 60
            assert config.data.continuation_weight == 25
            assert config.data.terminal_weight == 15

        # C variants: narrow domain
        if name == "sum_c":
            assert config.data.operand_min == -16 and config.data.operand_max == 16
            assert config.data.min_terms == 3 and config.data.max_terms == 3
        elif name == "neg_c":
            assert config.data.operand_min == -8 and config.data.operand_max == 8
        else:
            assert config.data.operand_min == -64 and config.data.operand_max == 64

        # NEG configs use 3-8 term range (needed for multi-operator validation)
        # SUM-C uses 3-3 (narrow domain), SUM-A/B use 3-8
        assert config.data.min_terms == 3


def test_plan_precommits_reproducible_fallback_ladder() -> None:
    plan = load_plan(PLAN_V2)
    fallback = plan["contingency"]

    assert fallback["calibration_split"] == "validation"
    assert fallback["calibration_evaluation_seed"] == 703000
    assert fallback["calibration_examples_per_operator"] == 128
    assert fallback["activation"]["final_data_may_trigger_tuning"] is False

    validation = fallback["validation"]
    assert validation["method"] == "leave_one_training_seed_out"
    assert validation["folds"] == 3
    assert validation["fitting_seeds_per_fold"] == 2
    assert validation["held_out_seeds_per_fold"] == 1
    assert validation["tune_per_seed"] is False
    assert validation["tune_per_operator"] is False
    assert validation["fit_statistics_on_fitting_seeds_only"] is True
    assert validation["held_out_seed_excluded_from_all_fit_statistics"] is True
    assert validation["paired_problem_set_across_seeds"] is True

    assert fallback["fixed_baselines"] == ["raw_sum", "bias_mean"]
    assert fallback["stage_order"] == [
        "global_shrinkage",
        "norm_controlled",
        "static_weighted_mean",
        "consensus_tempered",
    ]
    assert fallback["global_shrinkage"]["alpha_grid"] == [0.10, 0.20, 0.25, 0.50, 0.75, 1.00]

    norm = fallback["norm_controlled"]
    assert norm["center_each_position_over_vocabulary"] is True
    assert norm["equalization_target"] == "median_unit_rms"
    assert norm["epsilon"] == 1e-8
    assert norm["operator_specific_scales"] is False

    static = fallback["static_weighted_mean"]
    assert static["fitting_objective"] == "mean_gold_token_nll_plus_uniform_weight_l2"
    assert static["regularization_grid"] == [0.0, 0.01, 0.10, 1.0]
    assert static["optimizer"] == "deterministic_lbfgs"
    assert static["initialization"] == "uniform_weights"

    consensus = fallback["consensus_tempered"]
    assert consensus["weight_normalization"] == "sum_to_number_of_active_units"
    assert consensus["beta_zero_limit"] == "raw_sum_before_global_alpha"

    for family in ("norm_controlled", "static_weighted_mean", "consensus_tempered"):
        assert fallback[family]["alpha_grid_from"] == "global_shrinkage"

    selection = fallback["selection"]
    assert selection["choose_first_eligible_family"] is True
    assert selection["refit_selected_family_on_all_validation"] is True
    assert selection["refit_unselected_families"] is False
    assert selection["freeze_mixer_contract"] is True

    authorization = fallback["final_authorization"]
    assert authorization["path"] == "evaluations/fusion_calibration/final_authorization.json"
    assert authorization["abi_version"] == 1
    assert authorization["required_before_final_splits"] is True
    assert authorization["require_current_plan_hash"] is True
    assert authorization["require_current_git_commit"] is True
    assert authorization["require_experiment_fingerprint"] is True
    assert authorization["require_all_production_seeds"] is True

    assert fallback["learned_router_followup"]["part_of_confirmatory_final"] is False
    assert fallback["learned_router_followup"]["requires_new_plan_and_output_roots"] is True
    assert plan["reporting"]["preserve_raw_result_as_confirmatory"] is True
