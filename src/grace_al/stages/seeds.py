"""seeds (round 0): the initial structures.

A seeds file that exists is used as it is. Otherwise the generator named by
system.generator (surface, bulk, file, or a plug-in) builds it from `system:`.
"""

from __future__ import annotations

import logging
from collections import Counter

from grace_al import data
from grace_al.stages.base import RoundContext, StageResult

log = logging.getLogger("grace_al.stages.seeds")

KNOBS = {"surface": "system.initial.coverages / n_per_coverage / n_rattle, or more adsorbate "
                    "sites",
         "bulk": "system.initial.n_per_coverage / n_rattle, system.bulk.size",
         "file": "more structures in the seeds file"}


def build(cfg) -> list:
    from grace_al.systems.base import get_generator

    assert cfg.system is not None
    return get_generator(cfg.system.generator).build(cfg)


def summary(frames, rp_dim: int, generator: str | None = None) -> dict:
    """Atoms per element, frames per config type, and the elements too scarce for an
    artifact with rp_dim (round 0 needs rp_dim + 4 atoms of every element)."""
    elements: Counter[str] = Counter()
    configs: Counter[str] = Counter()
    for at in frames:
        elements.update(at.get_chemical_symbols())
        configs[str(at.info.get("config", "?")).split("_")[0]] += 1
    scarce = {el: n for el, n in elements.items() if n < rp_dim + 4}
    out = {"n_seeds": len(frames), "elements": dict(sorted(elements.items())),
           "configs": dict(configs), "scarce": scarce}
    if scarce:
        out["warning"] = (f"elements with fewer than artifact.rp_dim_round0 + 4 = {rp_dim + 4} "
                          f"atoms: {scarce}; rp_dim will be lowered. More: "
                          f"{KNOBS.get(generator or '', 'more seed structures')}")
    return out


class SeedsStage:
    name = "seeds"
    depends_on = ("system",)

    def applies(self, ctx: RoundContext) -> bool:
        return ctx.k == 0

    def run(self, ctx: RoundContext) -> StageResult:
        from ase.io import write

        cfg, source = ctx.cfg, "file"
        if not cfg.seeds.exists():
            if cfg.system is None:
                return StageResult("failed", {"error": f"{cfg.seeds} not found and no "
                                              "`system:` section to build it from"})
            log.info("building seeds (%s) -> %s", cfg.system.generator, cfg.seeds)
            write(cfg.seeds, build(cfg), format="extxyz")
            source = cfg.system.generator
        frames = data.read_frames(cfg.seeds)
        s = summary(frames, cfg.artifact.rp_dim_round0,
                    cfg.system.generator if cfg.system else "file")
        log.info("seeds: %d structures, atoms per element %s", s["n_seeds"], s["elements"])
        if "warning" in s:
            log.warning("seeds: %s", s["warning"])
        return StageResult("done", {**s, "source": source})
