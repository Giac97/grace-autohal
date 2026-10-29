"""artifact: the CALM artifact of round k = the driver's feature space, fitted on the data
we already have (round 0: the seeds, with placeholder labels).

A non-degenerate covariance needs at least rp_dim + 4 atoms of every element. Adsorbate
elements are few (round 0: seeds only), so rp_dim is capped by the scarcest element.
"""

from __future__ import annotations

import logging
import shutil

import pandas as pd

from grace_al import _tp, data
from grace_al.models import fit_dir
from grace_al.stages.base import RoundContext, StageResult

log = logging.getLogger("grace_al.stages.artifact")

MIN_RP_DIM = 8


class TooFewAtoms(ValueError):
    pass


def capped_rp_dim(rp_max: int, counts: dict[str, int]) -> int:
    """min(rp_max, scarcest element - 4); TooFewAtoms below MIN_RP_DIM."""
    rp_dim = min(rp_max, min(counts.values()) - 4)
    if rp_dim < MIN_RP_DIM:
        raise TooFewAtoms(f"too few atoms per element for a CALM artifact: {counts} (need "
                          f">= {MIN_RP_DIM + 4} of each); add seeds (adsorbate configurations)")
    return rp_dim


def build_params(cfg, k: int, counts: dict[str, int]) -> dict:
    """rp_dim and n_clusters of round k's artifact (round 0: rp_dim_round0, one cluster)."""
    a = cfg.artifact
    rp_max = a.rp_dim_round0 if k == 0 else a.rp_dim
    return {"rp_dim": capped_rp_dim(rp_max, counts), "rp_max": rp_max,
            "n_clusters": [1] if k == 0 else list(a.n_clusters)}


class ArtifactStage:
    name = "artifact"
    depends_on = ("artifact", "foundation_model", "fm_rounds")

    def applies(self, ctx: RoundContext) -> bool:
        return True

    def run(self, ctx: RoundContext) -> StageResult:
        path = ctx.rdir / "artifact_data.pkl.gz"
        if path.exists():
            df = pd.read_pickle(path)
        else:
            df = data.training_set(ctx.cfg, ctx.k)
            if df is None:
                df = data.placeholder_frame(data.read_frames(ctx.cfg.seeds))
            df.to_pickle(path)
        counts = data.element_counts(df)
        try:
            p = build_params(ctx.cfg, ctx.k, counts)
        except TooFewAtoms as err:
            return StageResult("failed", {"error": str(err), "element_counts": counts})
        if p["rp_dim"] < p["rp_max"]:
            log.info("round %d: rp_dim lowered to %d (atoms per element: %s)",
                     ctx.k, p["rp_dim"], counts)
        uq = ctx.rdir / "UQ"
        out = uq / "gmm_artifacts.npz"
        reused = self._reuse(ctx, df, p)
        if reused:
            log.info("round %d: artifact = the one built after round %d's fit", ctx.k, ctx.k - 1)
            if uq.exists():
                shutil.rmtree(uq)
            shutil.copytree(reused.parent, uq)
        else:
            _tp.uq_build(ctx.driver, [path], out, p["rp_dim"], p["n_clusters"],
                         n_workers=ctx.cfg.artifact.n_workers, log_file=uq / "build.log")
        info_text, info = _tp.uq_info(out)
        (uq / "info.txt").write_text(info_text)
        return StageResult("done", {"n_structures": len(df), "element_counts": counts,
                                    "rp_dim": p["rp_dim"], "n_clusters": p["n_clusters"],
                                    "reused_from_fit": bool(reused), "info": info})

    @staticmethod
    def _reuse(ctx: RoundContext, df, p: dict):
        """The previous fit's artifact, if it was built for this driver from the same data
        with the same parameters (finetune stores them in fit/UQ/params.json)."""
        import json

        if ctx.driver.kind != "finetuned" or ctx.driver.artifact is None:
            return None
        params = fit_dir(ctx.cfg, ctx.k - 1) / "UQ" / "params.json"
        if not params.exists():
            return None
        built = json.loads(params.read_text())
        want = {"rp_dim": p["rp_dim"], "n_clusters": p["n_clusters"],
                "dataset": data.dataset_hash(df)}
        return ctx.driver.artifact if built == want else None
