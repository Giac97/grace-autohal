import numpy as np
import pytest
from ase.build import fcc111

from grace_al.config.schema import AdsorbateConfig
from grace_al.systems.adsorbates import (
    lateral_distance,
    make_adsorbate,
    pick_sites,
    place,
    site_positions,
    slab_sites,
)


def placed(species, site="fcc"):
    slab = fcc111("Pd", (2, 2, 3), vacuum=8)
    mol, idx = make_adsorbate(AdsorbateConfig(species=species))
    at = place(slab, [(mol, idx, 1.5, (site, (0, 0), None))])
    return at[len(slab):], idx, slab


def test_co_binds_through_c_with_o_above():
    ads, idx, slab = placed("CO")
    assert ads[idx].symbol == "C"
    c, o = ads.positions[ads.get_chemical_symbols().index("C")], \
        ads.positions[ads.get_chemical_symbols().index("O")]
    assert o[2] > c[2] and np.allclose(o[:2], c[:2], atol=1e-6)
    assert c[2] == pytest.approx(slab.positions[:, 2].max() + 1.5)


def test_oh_binds_through_o():
    ads, idx, _ = placed("OH")
    assert ads[idx].symbol == "O"
    o, h = ads.positions
    assert h[2] > o[2]


def test_atom_and_bind_index_override():
    mol, idx = make_adsorbate(AdsorbateConfig(species="O"))
    assert len(mol) == 1 and idx == 0
    mol, idx = make_adsorbate(AdsorbateConfig(species="CO", bind_index=0))
    assert mol[idx].symbol == mol[0].symbol


def test_site_enumeration_and_minimum_image():
    slab = fcc111("Pd", (3, 3, 4), vacuum=8)
    assert set(slab_sites(slab)) == {"ontop", "bridge", "fcc", "hcp"}
    cand = site_positions(slab, ["ontop"], (3, 3))
    assert len(cand) == 9
    a = slab.cell.lengths()[0] / 3                       # nearest-neighbour distance
    p0 = cand[0][2]
    far = p0 + slab.cell[0, :2] * (2 / 3)                # two cells right = one cell left
    assert lateral_distance(p0, far, slab.cell) == pytest.approx(a, abs=1e-6)


def test_pick_sites_distance_or_none():
    slab = fcc111("Pd", (3, 3, 4), vacuum=8)
    cand = site_positions(slab, ["ontop"], (3, 3))
    rng = np.random.default_rng(0)
    pick = pick_sites(rng, cand, 3, 4.0, slab.cell)
    assert pick is not None
    assert pick_sites(rng, cand, 9, 4.0, slab.cell, tries=50) is None
