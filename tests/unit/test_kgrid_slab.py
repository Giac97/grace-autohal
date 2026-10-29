import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111

from grace_al.config.schema import Config
from grace_al.labelers.aims import aims_parameters, is_slab, k_grid, write_inputs
from grace_al.systems.base import SystemInfo, vacuum_gap

SLAB = SystemInfo("slab", ("O", "Pd"), 2)
BULK = SystemInfo("bulk", ("Al",))


def cfg(tmp_path, **aims):
    sd = tmp_path / "species"
    sd.mkdir(exist_ok=True)
    for z, el in ((8, "O"), (13, "Al"), (46, "Pd")):
        (sd / f"{z:02d}_{el}_default").write_text(f"  species {el}\n  nucleus {z}\n")
    return Config(home=tmp_path, label={"labeler": "aims"},
                  aims={"species_dir": "species", **aims})


# ---------------------------------------------------------------- k-grid
def test_kgrid_bulk():
    at = bulk("Al", "fcc", a=4.05, cubic=True)        # |b| = 2π/4.05 = 1.551 1/Å
    assert k_grid(at, 0.2) == [8, 8, 8]
    assert k_grid(at, 0.5) == [4, 4, 4]


def test_kgrid_slab_kz_one():
    at = fcc111("Pd", (3, 3, 5), vacuum=8)
    k = k_grid(at, 0.2, slab=True)
    assert k[2] == 1 and k[0] == k[1] > 1
    assert k_grid(at, 0.2)[2] > 1


def test_kgrid_tilted_cell():
    at = Atoms("Al", cell=[[3.0, 0, 0], [1.5, 2.6, 0], [0, 0, 5.0]], pbc=True)
    rec = np.linalg.norm(at.cell.reciprocal(), axis=1) * 2 * np.pi
    assert k_grid(at, 0.3) == [int(np.ceil(b / 0.3)) for b in rec]
    assert k_grid(at, 0.3)[0] == k_grid(at, 0.3)[1]   # hexagonal: a* = b*


# ---------------------------------------------------------------- slab detection
def test_vacuum_gap_with_wrapped_atoms():
    at = fcc111("Pd", (2, 2, 3), vacuum=7.5)
    gap = vacuum_gap(at)
    at.positions[:, 2] += at.cell[2, 2] * 0.8        # slab now crosses the cell boundary
    assert vacuum_gap(at) == pytest.approx(gap)
    assert gap == pytest.approx(15.0, abs=0.01)


def test_slab_detection():
    slab, b = fcc111("Pd", (2, 2, 3), vacuum=8), bulk("Al", cubic=True)
    assert is_slab(slab, None, 6.0)
    assert not is_slab(b, None, 6.0)
    assert is_slab(fcc111("Pd", (2, 2, 3), vacuum=2), SLAB, 6.0)     # SystemInfo says so
    cluster = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[10] * 3, pbc=False)
    assert not is_slab(cluster, SLAB, 6.0)
    full = fcc111("Pd", (2, 2, 3), vacuum=8, periodic=True)
    assert is_slab(full, None, 6.0)


# ---------------------------------------------------------------- parameters, control.in
def test_parameters_slab_vs_bulk_vs_cluster(tmp_path):
    c = cfg(tmp_path)
    p = aims_parameters(c, fcc111("Pd", (2, 2, 3), vacuum=8), SLAB)
    assert p["k_grid"][2] == 1 and p["use_dipole_correction"] is True
    assert p["compute_forces"] and p["compute_analytical_stress"]
    p = aims_parameters(c, bulk("Al", cubic=True), BULK)
    assert "use_dipole_correction" not in p and p["k_grid"] == [8, 8, 8]
    cluster = Atoms("Pd2", positions=[[0, 0, 0], [2.5, 0, 0]], pbc=False)
    p = aims_parameters(c, cluster)
    assert not {"k_grid", "use_dipole_correction", "compute_analytical_stress"} & set(p)
    assert p["compute_forces"]


def test_control_in_slab(tmp_path):
    c = cfg(tmp_path, kspacing=0.3, parameters={"elsi_build_dm_method": "pdgemm"})
    at = fcc111("Pd", (2, 2, 3), vacuum=8)
    at.info["gamma"] = 3.0                              # HAL leftovers are not written
    write_inputs(c, at, tmp_path / "t", SLAB)
    control = (tmp_path / "t" / "control.in").read_text()
    kz = k_grid(at, 0.3, slab=True)
    assert f"k_grid                             {kz[0]} {kz[1]} 1" in control \
        or f"k_grid {kz[0]} {kz[1]} 1" in " ".join(control.split())
    words = " ".join(control.split())
    assert "use_dipole_correction" in words and "compute_analytical_stress .true." in words
    assert "compute_forces .true." in words and "elsi_build_dm_method pdgemm" in words
    assert "occupation_type methfessel-paxton 0.1 1" in words
    assert "species Pd" in control
    geo = (tmp_path / "t" / "geometry.in").read_text()
    assert geo.count("lattice_vector") == 3 and geo.count("atom_frac") == 12


def test_control_in_bulk_no_dipole(tmp_path):
    write_inputs(cfg(tmp_path), bulk("Al", cubic=True), tmp_path / "t", BULK)
    words = " ".join((tmp_path / "t" / "control.in").read_text().split())
    assert "use_dipole_correction" not in words and "k_grid 8 8 8" in words


def test_parameter_null_removes_default(tmp_path):
    p = aims_parameters(cfg(tmp_path, parameters={"use_dipole_correction": None}),
                        fcc111("Pd", (2, 2, 3), vacuum=8), SLAB)
    assert "use_dipole_correction" not in p
