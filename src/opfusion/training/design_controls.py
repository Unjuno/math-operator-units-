from __future__ import annotations

from dataclasses import replace

from .data import SyntheticTraceFactory, TrainingExample
from .design_config import model_design


_ORIGINAL_TRAINING_EXAMPLE = SyntheticTraceFactory.training_example


def training_example_with_model_design(
    self: SyntheticTraceFactory,
    job_id: str,
    *,
    seed: int,
    split: str,
    step: int,
    sample_index: int,
    forced_operator: str | None = None,
) -> TrainingExample:
    design = model_design(self.config)
    if job_id != "base.common" or design.base_target_mode == "identity":
        return _ORIGINAL_TRAINING_EXAMPLE(
            self,
            job_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index,
            forced_operator=forced_operator,
        )

    operator_id = forced_operator or self.joint_operator(
        seed=seed,
        split=split,
        step=step,
        sample_index=sample_index,
        namespace="base",
    )
    # Rejection sampling is deterministic and preserves the underlying stable
    # IID partition. The weak base receives real operator targets only on a
    # deliberately restricted operand/length domain.
    for attempt in range(self.config.max_partition_attempts):
        candidate = _ORIGINAL_TRAINING_EXAMPLE(
            self,
            operator_id,
            seed=seed,
            split=split,
            step=step,
            sample_index=sample_index + attempt * 1_000_003,
            forced_operator=operator_id,
        )
        if (
            len(candidate.initial_values) <= design.base_weak_max_terms
            and all(abs(value) <= design.base_weak_operand_abs_max for value in candidate.initial_values)
        ):
            return replace(
                candidate,
                job_id="base.common",
                task=f"weak_multitask:{candidate.task}",
            )
    raise RuntimeError(
        "failed to sample a weak-multitask base example within the declared "
        f"operand/length limits for operator={operator_id}, split={split}"
    )


def install_model_design_controls() -> None:
    SyntheticTraceFactory.training_example = training_example_with_model_design  # type: ignore[method-assign]
