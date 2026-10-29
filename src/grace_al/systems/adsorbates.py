"""Adsorbates: molecule + binding atom, adsorption sites, placement with a minimum distance."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase import build as ase_build
from ase.data import atomic_numbers

if TYPE_CHECKING:
    from grace_al.config.schema import AdsorbateConfig

# Pauling electronegativity: a molecule binds through its least electronegative heavy atom
# (CO -> C, OH -> O, NH3 -> N); override with adsorbate.bind_index
ELECTRONEGATIVITY = {"B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Si": 1.90,
                     "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66}

Site = tuple[str, tuple[int, int], np.ndarray]     # (site name, offset, lateral position)


def make_adsorbate(a: AdsorbateConfig) -> tuple[Atoms, int]:
    """(molecule, index of the binding atom); the rest of the molecule points along +z."""
    name = a.species
    if name in atomic_numbers:
        return Atoms(name), 0
    mol = ase_build.molecule(name)
    idx = a.bind_index
    if idx is None:
        heavy = [i for i, s in enumerate(mol.get_chemical_symbols()) if s != "H"]
        idx = min(heavy, key=lambda i: ELECTRONEGATIVITY.get(mol[i].symbol, 9.0))
    rest = [i for i in range(len(mol)) if i != idx]
    v = mol.positions[rest].mean(axis=0) - mol.positions[idx]
    if np.linalg.norm(v) > 1e-6:
        mol.rotate(v, (0, 0, 1), center=mol.positions[idx])
    return mol, idx


def slab_sites(slab: Atoms) -> list[str]:
    return list(slab.info["adsorbate_info"]["sites"])


def site_positions(slab: Atoms, sites: list[str], size) -> list[Site]:
    """Every (site, offset) of the supercell with its lateral position (probe atom)."""
    out = []
    for site, i, j in itertools.product(sites, range(size[0]), range(size[1])):
        probe = slab.copy()
        ase_build.add_adsorbate(probe, "X", 1.0, site, offset=(i, j))
        out.append((site, (i, j), probe.positions[-1, :2]))
    return out


def lateral_distance(p, q, cell) -> float:
    """In-plane distance with the 2D minimum image."""
    d = np.array([p[0] - q[0], p[1] - q[1], 0.0])
    frac = np.linalg.solve(np.asarray(cell).T, d)
    frac[:2] -= np.round(frac[:2])
    return float(np.linalg.norm(np.asarray(cell).T @ frac))


def place(slab: Atoms, items) -> Atoms:
    """items: (molecule, bind index, height, (site, offset, _)) per adsorbate."""
    at = slab.copy()
    for mol, idx, height, (site, offset, _) in items:
        ase_build.add_adsorbate(at, mol, height, site, offset=offset, mol_index=idx)
    return at


def pick_sites(rng: np.random.Generator, candidates: list[Site], n: int, d_min: float,
               cell, tries: int = 1000) -> list[Site] | None:
    """n distinct candidates at least d_min apart (laterally), or None."""
    for _ in range(tries):
        pick = [candidates[i] for i in rng.choice(len(candidates), n, replace=False)]
        if all(lateral_distance(p[2], q[2], cell) >= d_min
               for p, q in itertools.combinations(pick, 2)):
            return pick
    return None
