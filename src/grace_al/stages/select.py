"""select: grace_uq select on the walkers' candidates against round k's artifact.

Strategy: calm-greedy:{max_per_round} (stops at its natural set) + random-all:{n_random}
(an unbiased sample, used by validate). Output: selected.pkl.gz (as written by grace_uq,
with the `strategy` column) and selected.extxyz (strategy in atoms.info), without picks
identical to a start structure (starts.extxyz: seeds and earlier picks).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from grace_al import _tp, data
from grace_al.stages.base import RoundContext, StageResult

log = logging.getLogger("grace_al.stages.select")


def walker_dirs(hal: Path) -> list[Path]:
    """Walker folders with candidates: hal/walkers.txt (explore stage) if present, else
    every folder below hal/ that holds a hal_features.npz."""
    listing = hal / "walkers.txt"
    if listing.exists():
        dirs = [hal / line.strip() if not Path(line.strip()).is_absolute()
                else Path(line.strip()) for line in listing.read_text().splitlines()
                if line.strip()]
    else:
        dirs = sorted(p.parent for p in hal.rglob("hal_features.npz"))
    return [d for d in dirs if (d / "hal_features.npz").exists()]


def strategy(cfg) -> list[str]:
    s = [f"calm-greedy:{cfg.select.max_per_round}"]
    if cfg.select.n_random:
        s.append(f"random-all:{cfg.select.n_random}")
    return s


class SelectStage:
    name = "select"
    depends_on = ("select",)

    def applies(self, ctx: RoundContext) -> bool:
        return True

    def run(self, ctx: RoundContext) -> StageResult:
        from ase.io import write

        pkl, xyz = ctx.rdir / "selected.pkl.gz", ctx.rdir / "selected.extxyz"
        walkers = walker_dirs(ctx.rdir / "hal")
        df, parsed = pd.DataFrame(), {"natural": True, "natural_set": 0}
        if walkers:
            logf = ctx.rdir / "select.log"
            logf.unlink(missing_ok=True)
            _tp.uq_select(walkers, ctx.rdir / "UQ" / "gmm_artifacts.npz", strategy(ctx.cfg),
                          pkl, log_file=logf)
            df = pd.read_pickle(pkl)
            parsed = _tp.parse_select_log(logf.read_text(errors="ignore"))
        else:
            log.warning("round %d: no walker produced candidates; nothing to select", ctx.k)
        # HAL collects step 0 when the start is extrapolative: that frame is a start
        # structure, i.e. a seed or a previous pick, labeled already or in this round
        starts = ctx.rdir / "starts.extxyz"
        known = ({data.structure_hash(at, 4) for at in data.read_frames(starts)}
                 if starts.exists() else set())
        frames, dropped = [], 0
        for _, row in df.iterrows():
            at = row["ase_atoms"].copy()
            at.calc = None
            if data.structure_hash(at, 4) in known:
                dropped += 1
                continue
            at.info["strategy"] = row.get("strategy", "")
            frames.append(at)
        if dropped:
            log.info("round %d: %d picks dropped: identical to a start structure", ctx.k,
                     dropped)
        if frames:
            write(xyz, frames, format="extxyz")
        else:
            xyz.write_text("")

        picks = dict(Counter(at.info["strategy"] for at in frames))
        n_greedy = int(picks.get("calm-greedy", 0))
        natural_set = parsed["natural_set"] if parsed["natural_set"] is not None else n_greedy
        summary = {"n_walkers": len(walkers), "n_selected": len(frames), "picks": picks,
                   "dropped_known": dropped,
                   "natural": parsed["natural"], "natural_set": natural_set}
        log.info("round %d: %d structures selected %s", ctx.k, len(frames), picks)
        conv = ctx.cfg.convergence
        if ctx.k > 0 and parsed["natural"] and natural_set <= conv.max_natural_set:
            log.info("round %d: converged (natural set %d <= max_natural_set %d)",
                     ctx.k, natural_set, conv.max_natural_set)
            return StageResult("converged", {**summary, "converged": "natural_set"})
        return StageResult("done", summary)
