"""finetune: gracemaker on all labeled data so far, then the CALM artifact of the new model.

input.yaml comes from finetune_template.yaml (frozen `reducing_` weights by 1L/2L/3L,
reference_energy auto, shift auto, cosine LR down to lr/12); finetune.extra is deep-merged
into it. An interrupted fit (checkpoint, no FINISHED) continues with `gracemaker -rl`.

start_from:
  fm        fine-tune the foundation model from scratch every round (the default)
  previous  continue from round k-1's model: potential.filename/checkpoint_name point to
            it, fit.reset_epoch_and_step restarts the LR schedule, and reference_energy
            `auto` is replaced by the E0 that round k-1's fit computed (from its log.txt),
            so the model's energy zero stays where it was trained
"""

from __future__ import annotations

import json
import logging
from importlib import resources

import jinja2
import pandas as pd
import yaml

from grace_al import _tp, data
from grace_al.models import fit_dir, fitted
from grace_al.stages.artifact import TooFewAtoms, capped_rp_dim
from grace_al.stages.base import RoundContext, StageResult
from grace_al.utils import deep_merge

log = logging.getLogger("grace_al.stages.finetune")


def has_stress(df: pd.DataFrame) -> bool:
    return "stress" in df.columns and bool(df["stress"].notna().all())


def render_input(cfg, train, stress: bool, start_from: str = "fm", previous=None,
                 reference_energy=None) -> dict:
    """input.yaml of a fine-tune as a dict (template + finetune.extra)."""
    f = cfg.finetune
    text = (resources.files("grace_al.stages") / "finetune_template.yaml").read_text()
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    rendered = env.from_string(text).render(
        f=f, train=str(train), fm=cfg.foundation_model, stress=stress,
        start_from=start_from, previous=previous and {
            "model_yaml": str(previous.model_yaml), "checkpoint": str(previous.checkpoint)},
        reference_energy=f.reference_energy if reference_energy is None else reference_energy,
        trainable=_tp.finetune_trainable_variables(cfg.foundation_model) if f.frozen else None)
    return deep_merge(yaml.safe_load(rendered), f.extra)


class FinetuneStage:
    name = "finetune"
    depends_on = ("finetune", "foundation_model")

    def applies(self, ctx: RoundContext) -> bool:
        return True

    def run(self, ctx: RoundContext) -> StageResult:
        fit = ctx.rdir / "fit"
        if (fit / "FINISHED").exists():
            log.info("round %d: fit finished before; building its artifact", ctx.k)
        elif (fit / "seed" / "1" / "checkpoints" / "checkpoint.index").exists():
            log.info("round %d: continuing the interrupted fit (gracemaker -rl)", ctx.k)
            _tp.gracemaker(fit, restart_latest=True)
            (fit / "FINISHED").touch()
        else:
            self._write_input(ctx, fit)
            _tp.gracemaker(fit)
            (fit / "FINISHED").touch()
        train = pd.read_pickle(fit / "train.pkl.gz")
        potential = yaml.safe_load((fit / "input.yaml").read_text())["potential"]
        return StageResult("done", {
            "n_train": len(train), "stress": has_stress(train),
            "start_from": "previous" if "filename" in potential else "fm",
            "reference_energies": _tp.fit_reference_energies(fit),
            "artifact": self._build_artifact(ctx, train)})

    @staticmethod
    def _write_input(ctx: RoundContext, fit) -> None:
        fit.mkdir(parents=True, exist_ok=True)
        df = data.training_set(ctx.cfg, ctx.k + 1)
        if df is None:
            raise RuntimeError(f"round {ctx.k}: no labeled data to fine-tune on")
        train = fit / "train.pkl.gz"
        df.to_pickle(train)
        start_from, previous, ref = ctx.cfg.finetune.start_from, None, None
        if start_from == "previous":
            if ctx.k == 0 or not (fit_dir(ctx.cfg, ctx.k - 1) / "FINISHED").exists():
                log.info("round %d: no previous fit; starting from the foundation model", ctx.k)
                start_from = "fm"
            else:
                previous = fitted(ctx.cfg, ctx.k - 1)
                if ctx.cfg.finetune.reference_energy == "auto":
                    ref = _tp.fit_reference_energies(fit_dir(ctx.cfg, ctx.k - 1))
                    if ref is None:
                        raise RuntimeError(
                            f"start_from previous: no E0 in {fit_dir(ctx.cfg, ctx.k - 1)}"
                            "/log.txt; set finetune.reference_energy explicitly")
        inp = render_input(ctx.cfg, train, has_stress(df), start_from, previous, ref)
        (fit / "input.yaml").write_text(yaml.safe_dump(inp, sort_keys=False))

    @staticmethod
    def _build_artifact(ctx: RoundContext, train: pd.DataFrame) -> dict | None:
        """fit/UQ/gmm_artifacts.npz for the new model, with the parameters the next round's
        artifact stage will use, so it can reuse this one (params.json)."""
        counts = data.element_counts(train)
        try:
            rp_dim = capped_rp_dim(ctx.cfg.artifact.rp_dim, counts)
        except TooFewAtoms as err:
            log.warning("round %d: no artifact for the fitted model: %s", ctx.k, err)
            return None
        uq = ctx.rdir / "fit" / "UQ"
        out = uq / "gmm_artifacts.npz"
        params: dict = {"rp_dim": rp_dim, "n_clusters": list(ctx.cfg.artifact.n_clusters),
                  "dataset": data.dataset_hash(train)}
        p = uq / "params.json"
        if not (out.exists() and p.exists() and json.loads(p.read_text()) == params):
            model = fitted(ctx.cfg, ctx.k)
            _tp.uq_build(model, [ctx.rdir / "fit" / "train.pkl.gz"], out, rp_dim,
                         params["n_clusters"], n_workers=ctx.cfg.artifact.n_workers,
                         log_file=uq / "build.log")
            p.write_text(json.dumps(params) + "\n")
        return {"rp_dim": rp_dim, "n_clusters": params["n_clusters"]}
