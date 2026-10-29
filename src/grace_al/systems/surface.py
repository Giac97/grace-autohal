"""Surface + adsorbate seeds (generator `surface`).

Configurations, each optionally pre-relaxed with the foundation model (bottom layers fixed)
and then copied with random displacements:
  - the clean slab at every lateral strain                 clean_strain0.98
  - one adsorbate on every site type, at every strain      CO_fcc_strain1
  - several adsorbates on random distinct sites (coverage) COx2_0 (mix: CO+Ox3_1)
  - rattled copies of all of them                          COx2_0_rattle0

All cells are periodic in 3D (vacuum along z), as HAL and the aims k-grid need. The random
draws are those of the original build_seeds.py, so a single-species config gives the same
structures for the same `initial.seed`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ase import build as ase_build

from grace_al.systems.adsorbates import (
    make_adsorbate,
    pick_sites,
    place,
    site_positions,
    slab_sites,
)
from grace_al.systems.base import SystemInfo

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import AdsorbateConfig, Config, SurfaceConfig

log = logging.getLogger("grace_al.systems.surface")


def make_slab(s: SurfaceConfig) -> Atoms:
    kw: dict = {"size": tuple(s.size), "vacuum": s.vacuum}
    if s.a:
        kw["a"] = s.a
    if s.c:
        kw["c"] = s.c
    if s.orthogonal is not None:
        kw["orthogonal"] = s.orthogonal
    slab = getattr(ase_build, s.facet)(s.element, **kw)
    slab.pbc = True
    return slab


def strain_lateral(atoms: Atoms, f: float) -> Atoms:
    """Scale the in-plane cell vectors (a, b) by f; z and the vacuum stay."""
    at = atoms.copy()
    cell = at.cell.array.copy()
    cell[:2] *= f
    at.set_cell(cell, scale_atoms=True)
    return at


def adsorbates(cfg: Config) -> list[AdsorbateConfig]:
    assert cfg.system is not None
    a = cfg.system.adsorbate
    return a if isinstance(a, list) else [a] if a else []


def configs(cfg: Config, rng: np.random.Generator | None = None) -> list[Atoms]:
    """The unrelaxed, unrattled configurations (info: config, generator). The coverage
    draws come from `rng` (default: a new one from initial.seed)."""
    assert cfg.system is not None and cfg.system.surface is not None
    s, init = cfg.system.surface, cfg.system.initial
    rng = rng or np.random.default_rng(init.seed)
    slab0 = make_slab(s)
    frames = []
    for f in init.strains:
        frames.append(_tag(strain_lateral(slab0, f), f"clean_strain{f:g}"))
    ads = [(a, *make_adsorbate(a)) for a in adsorbates(cfg)]
    for a, mol, idx in ads:
        for site in a.sites or slab_sites(slab0):
            for f in init.strains:
                at = place(slab0, [(mol, idx, a.height, (site, (0, 0), None))])
                frames.append(_tag(strain_lateral(at, f), f"{a.species}_{site}_strain{f:g}"))
    groups = [ads] if init.mix and len(ads) > 1 else [[x] for x in ads]
    for group in groups:
        frames += _coverages(rng, slab0, s, init, group)
    return frames


def _coverages(rng, slab0, s, init, group) -> list[Atoms]:
    """Random coverage configs of one species (or a mix of several, chosen per site)."""
    if len(group) == 1:
        sites = group[0][0].sites or slab_sites(slab0)
    else:
        sites = [x for x in slab_sites(slab0)
                 if any(x in (a.sites or slab_sites(slab0)) for a, *_ in group)]
    candidates = site_positions(slab0, sites, s.size)
    d_min = max(a.min_distance for a, *_ in group)
    out = []
    for n in init.coverages:
        for r in range(init.n_per_coverage):
            pick = pick_sites(rng, candidates, n, d_min, slab0.cell)
            if pick is None:
                log.warning("no %d sites %.2f Å apart: coverage %d skipped", n, d_min, n)
                break
            if len(group) == 1:
                chosen = [group[0]] * n
                name = group[0][0].species
            else:
                chosen = [group[i] for i in rng.choice(len(group), n)]
                name = "+".join(sorted({a.species for a, *_ in chosen}))
            at = place(slab0, [(mol, idx, a.height, p)
                               for (a, mol, idx), p in zip(chosen, pick, strict=True)])
            out.append(_tag(at, f"{name}x{n}_{r}"))
    return out


def _tag(at: Atoms, config: str) -> Atoms:
    at.info = {"config": config, "generator": "surface"}
    return at


def rattle(frames: list[Atoms], rng, n_rattle: int, std: float) -> list[Atoms]:
    """Each frame followed by n_rattle copies with Gaussian displacements."""
    out = []
    for at in frames:
        at.info.pop("adsorbate_info", None)
        out.append(at)
        for r in range(n_rattle):
            rat = at.copy()
            rat.positions += rng.normal(0.0, std, rat.positions.shape)
            rat.info["config"] = f"{at.info['config']}_rattle{r}"
            out.append(rat)
    return out


class SurfaceGenerator:
    name = "surface"

    def build(self, cfg: Config) -> list[Atoms]:
        from grace_al.systems.relax import fm_relax

        assert cfg.system is not None and cfg.system.surface is not None
        init = cfg.system.initial
        # the coverage draws and the rattling share one generator, as in build_seeds.py
        rng = np.random.default_rng(init.seed)
        frames = configs(cfg, rng)
        if init.fm_relax:
            frames = fm_relax(frames, cfg.foundation_model, cfg.system.surface.fix_layers,
                              init.fmax, init.relax_steps)
        return rattle(frames, rng, init.n_rattle, init.rattle_std)

    def info(self, cfg: Config) -> SystemInfo:
        assert cfg.system is not None and cfg.system.surface is not None
        elements = {cfg.system.surface.element}
        for a in adsorbates(cfg):
            elements |= set(make_adsorbate(a)[0].get_chemical_symbols())
        return SystemInfo("slab", tuple(sorted(elements)), vacuum_axis=2)

