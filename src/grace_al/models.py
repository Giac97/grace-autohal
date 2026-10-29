"""Model references and the driver of each round."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from grace_al.config.schema import Config


@dataclass(frozen=True)
class ModelRef:
    kind: Literal["fm", "finetuned"]
    model_yaml: Path
    checkpoint: Path                      # TF checkpoint prefix, no .index
    artifact: Path | None = None          # CALM artifact (gmm_artifacts.npz) for this model


class MissingModel(RuntimeError):
    pass


def fit_dir(cfg: Config, k: int) -> Path:
    return cfg.workdir / f"round_{k:02d}" / "fit"


def fitted(cfg: Config, k: int) -> ModelRef:
    """The model fine-tuned in round k (with its artifact fit/UQ/..., if built)."""
    fit = fit_dir(cfg, k)
    seed = fit / "seed" / "1"
    artifact = fit / "UQ" / "gmm_artifacts.npz"
    return ModelRef("finetuned", seed / "model.yaml",
                    seed / "checkpoints" / "checkpoint.best_test_loss",
                    artifact if artifact.exists() else None)


def driver(cfg: Config, k: int) -> ModelRef:
    """The model that drives round k: the foundation model for k < fm_rounds, else the one
    fine-tuned in round k-1."""
    from grace_al import _tp

    if k < cfg.fm_rounds:
        return _tp.fm_ref(cfg.foundation_model)
    if not (fit_dir(cfg, k - 1) / "FINISHED").exists():
        raise MissingModel(f"round {k} is driven by the model fine-tuned in round {k - 1}, "
                           f"but {fit_dir(cfg, k - 1)} has no FINISHED: run round {k - 1} first")
    return fitted(cfg, k - 1)
