from pathlib import Path

import pytest

from grace_al import _tp
from grace_al.config.schema import Config
from grace_al.models import ModelRef
from grace_al.stages.finetune import render_input

TRAIN = Path("/c/runs/round_00/fit/train.pkl.gz")


def cfg(tmp_path, fm="GRACE-2L-SMAX-OMAT-medium", **finetune):
    return Config(home=tmp_path, label={"labeler": "manual"}, foundation_model=fm,
                  finetune=finetune)


def original_input(fm, lr=1e-3, stress=False):
    """input.yaml as al_workflow.py wrote it (the defaults)."""
    loss = {"energy": {"weight": 16}, "forces": {"weight": 32}}
    if stress:
        loss["stress"] = {"weight": 128.0}
    return {
        "seed": 1, "cutoff": 6.0,
        "data": {"filename": str(TRAIN), "test_size": 0.1, "reference_energy": "auto"},
        "potential": {"finetune_foundation_model": fm, "reduce_elements": True,
                      "shift": "auto"},
        "fit": {"loss": loss, "optimizer": "Adam",
                "opt_params": {"learning_rate": lr, "use_ema": True, "ema_momentum": 0.99,
                               "weight_decay": None, "clipnorm": 1.0},
                "scheduler": "cosine_decay",
                "scheduler_params": {"minimum_learning_rate": lr / 12},
                "target_total_updates": 10000, "batch_size": 16, "eval_init_stats": True,
                "jit_compile": True,
                "trainable_variable_names": _tp.finetune_trainable_variables(fm)}}


@pytest.mark.parametrize("fm", ["GRACE-1L-OMAT", "GRACE-2L-SMAX-OMAT-medium",
                                "GRACE-3L-X"])
@pytest.mark.parametrize("stress", [False, True])
def test_same_as_original_scripts(tmp_path, fm, stress):
    assert render_input(cfg(tmp_path, fm), TRAIN, stress) == original_input(fm, stress=stress)


def test_trainable_lists_by_layers(tmp_path):
    get = lambda fm: render_input(cfg(tmp_path, fm), TRAIN, False)["fit"][
        "trainable_variable_names"]
    assert get("GRACE-1L-OMAT") == ["rho/reducing_"]
    assert get("GRACE-2L-OMAT") == ["I2/reducing_", "rho/reducing_", "I1/reducing_"]
    assert len(get("GRACE-3L-X")) == 5


def test_not_frozen_trains_everything(tmp_path):
    assert "trainable_variable_names" not in render_input(cfg(tmp_path, frozen=False), TRAIN,
                                                          False)["fit"]


def test_minimum_learning_rate_and_weights(tmp_path):
    inp = render_input(cfg(tmp_path, learning_rate=3e-3, force_weight=10, batch_size=8),
                       TRAIN, False)
    assert inp["fit"]["scheduler_params"]["minimum_learning_rate"] == pytest.approx(2.5e-4)
    assert inp["fit"]["loss"]["forces"] == {"weight": 10}
    assert inp["fit"]["batch_size"] == 8


def test_extra_deep_merged(tmp_path):
    inp = render_input(cfg(tmp_path, extra={"fit": {"opt_params": {"clipnorm": 0.5},
                                                    "batch_size": 4},
                                            "data": {"test_size": 0.2}}), TRAIN, False)
    assert inp["fit"]["opt_params"]["clipnorm"] == 0.5
    assert inp["fit"]["opt_params"]["learning_rate"] == 1e-3      # kept
    assert inp["fit"]["batch_size"] == 4
    assert inp["data"]["test_size"] == 0.2


def test_reference_energy_dict(tmp_path):
    inp = render_input(cfg(tmp_path, reference_energy={"Pd": -1.5, "O": -0.5}), TRAIN, False)
    assert inp["data"]["reference_energy"] == {"Pd": -1.5, "O": -0.5}


def test_start_from_previous(tmp_path):
    prev = ModelRef("finetuned", Path("/r0/fit/seed/1/model.yaml"),
                    Path("/r0/fit/seed/1/checkpoints/checkpoint.best_test_loss"))
    inp = render_input(cfg(tmp_path, start_from="previous"), TRAIN, False, "previous", prev,
                       reference_energy={"Pd": -136000.5})
    assert inp["potential"] == {"filename": str(prev.model_yaml),
                                "checkpoint_name": str(prev.checkpoint)}
    assert inp["fit"]["reset_epoch_and_step"] is True
    assert inp["data"]["reference_energy"] == {"Pd": -136000.5}
    assert "trainable_variable_names" in inp["fit"]
