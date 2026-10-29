import numpy as np
import pytest

from grace_al import parity
from grace_al.stages.validate import errors

SYM = [["Pd"] * 4 + ["O"], ["Pd"] * 4 + ["O", "O"], ["Pd"] * 4, ["Pd"] * 3 + ["O"]]


def test_element_offsets_recover_shifts():
    rng = np.random.default_rng(0)
    e_dft = rng.normal(size=4)
    true = {"O": -2047.0, "Pd": -135000.0}
    e_pred = parity.shift(e_dft, SYM, {k: -v for k, v in true.items()})   # pred = dft + n.d
    off = parity.element_offsets(e_dft, e_pred, SYM)
    assert off == pytest.approx(true)
    assert parity.shift(e_pred, SYM, off) == pytest.approx(e_dft)


def test_errors_independent_of_reference_energies():
    f = [np.zeros((len(s), 3)) for s in SYM]
    e_dft = np.array([1.0, 2.0, 3.0, 4.0])
    noise = np.array([0.01, -0.02, 0.0, 0.01])
    e_pred = parity.shift(e_dft + noise, SYM, {"O": 500.0, "Pd": -7.0})
    r = errors(e_dft, e_pred, SYM, f, [x + 0.1 for x in f])
    assert r["energy_rmse_per_atom"] < 0.01
    assert r["force_rmse"] == pytest.approx(0.1) and r["force_max"] == pytest.approx(0.1)


def test_plot_and_npz(tmp_path):
    rng = np.random.default_rng(1)
    f = [rng.normal(size=(len(s), 3)) for s in SYM]
    d = parity.energy_force_data(SYM, [1, 2, 3, 4], [1.1, 2.1, 2.9, 4.0], f,
                                 [x + 0.05 for x in f], ["seed", "calm-greedy", "seed", "x"])
    png = tmp_path / "parity.png"
    parity.plot_energy_forces(d, png, "test")
    npz = parity.save(png, d)
    assert png.stat().st_size > 5000
    back = np.load(npz)
    assert back["f_dft"].shape == (3 * sum(map(len, SYM)),)
    assert list(back["group"]) == ["seed", "calm-greedy", "seed", "x"]
    assert set(back["f_element"]) == {"Pd", "O"}
