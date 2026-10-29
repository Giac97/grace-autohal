"""Bulk seeds (generator `bulk`): a supercell at isotropic strains, with random vacancies
and substitutions, plus rattled copies.

  system:
    generator: bulk
    bulk: {element: Al, crystalstructure: fcc, size: [2, 2, 2], vacancies: [0, 1],
           substitutions: {Cu: 0.1}}
    initial: {strains: [0.98, 1.0, 1.02], n_per_coverage: 2, n_rattle: 1, fm_relax: false}

Configs: `bulk_strain1` (pristine), `vac1_strain1_0` (n_per_coverage random variants per
vacancy count and strain; substitutions are applied to every variant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grace_al.systems.base import SystemInfo
from grace_al.systems.surface import rattle

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import BulkConfig, Config


def make_bulk(b: BulkConfig) -> Atoms:
    from ase.build import bulk

    kw: dict = {"cubic": b.cubic}
    for k in ("crystalstructure", "a", "c"):
        if getattr(b, k) is not None:
            kw[k] = getattr(b, k)
    return bulk(b.element, **kw).repeat(tuple(b.size))


def strain_iso(at: Atoms, f: float) -> Atoms:
    at = at.copy()
    at.set_cell(at.cell.array * f, scale_atoms=True)
    return at


def variant(base: Atoms, rng, n_vac: int, subs: dict[str, float]) -> Atoms:
    at = base.copy()
    host = list(range(len(at)))
    for el, frac in subs.items():
        n = int(round(frac * len(at)))
        idx = rng.choice(host, n, replace=False)
        at.symbols[idx] = el
        host = [i for i in host if i not in set(idx)]
    if n_vac:
        del at[sorted(rng.choice(len(at), n_vac, replace=False), reverse=True)]
    return at


class BulkGenerator:
    name = "bulk"

    def configs(self, cfg: Config, rng) -> list[Atoms]:
        assert cfg.system is not None and cfg.system.bulk is not None
        b, init = cfg.system.bulk, cfg.system.initial
        base = make_bulk(b)
        frames = []
        for f in init.strains:
            for n_vac in b.vacancies:
                pristine = n_vac == 0 and not b.substitutions
                for r in range(1 if pristine else init.n_per_coverage):
                    at = strain_iso(variant(base, rng, n_vac, b.substitutions), f)
                    name = "bulk" if pristine else f"vac{n_vac}"
                    at.info = {"config": f"{name}_strain{f:g}" + ("" if pristine else f"_{r}"),
                               "generator": "bulk"}
                    at.pbc = True
                    frames.append(at)
        return frames

    def build(self, cfg: Config) -> list[Atoms]:
        from grace_al.systems.relax import fm_relax

        assert cfg.system is not None
        init = cfg.system.initial
        rng = np.random.default_rng(init.seed)
        frames = self.configs(cfg, rng)
        if init.fm_relax:
            frames = fm_relax(frames, cfg.foundation_model, 0, init.fmax, init.relax_steps)
        return rattle(frames, rng, init.n_rattle, init.rattle_std)

    def info(self, cfg: Config) -> SystemInfo:
        assert cfg.system is not None and cfg.system.bulk is not None
        b = cfg.system.bulk
        elements = set(make_bulk(b).get_chemical_symbols()) | set(b.substitutions)
        return SystemInfo("bulk", tuple(sorted(elements)))
