import numpy as np
import pandas as pd
from ase.build import bulk, fcc111
from ase.calculators.singlepoint import SinglePointCalculator

from grace_al import data
from grace_al.config.schema import Config


def labeled(at, e=-1.0, stress=True):
    at = at.copy()
    kw = {"energy": e, "forces": np.zeros((len(at), 3))}
    if stress:
        kw["stress"] = np.zeros(6)
    at.calc = SinglePointCalculator(at, **kw)
    return at


def test_to_dataframe_columns_and_no_calculator():
    df = data.to_dataframe([labeled(bulk("Al"), -3.0), labeled(bulk("Al", cubic=True), -12.0)])
    assert list(df.columns) == ["ase_atoms", "energy", "forces", "stress", "energy_corrected"]
    assert df["stress"].iloc[0].shape == (3, 3)
    assert (df["energy_corrected"] == df["energy"]).all()
    assert all(at.calc is None for at in df["ase_atoms"])


def test_to_dataframe_stress_only_if_all_have_it():
    df = data.to_dataframe([labeled(bulk("Al")), labeled(bulk("Al"), stress=False)])
    assert "stress" not in df.columns


def test_placeholder_frame_zero_labels():
    df = data.placeholder_frame([fcc111("Pd", (2, 2, 3), vacuum=5)])
    assert df["energy"].iloc[0] == 0.0 and df["forces"].iloc[0].shape == (12, 3)
    assert not df["forces"].iloc[0].any()


def test_structure_hash():
    a = bulk("Al", cubic=True)
    b = a.copy()
    b.positions[0] += 1e-8
    c = a.copy()
    c.positions[0] += 1e-3
    assert data.structure_hash(a) == data.structure_hash(b) != data.structure_hash(c)
    assert data.structure_hash(a, 2) == data.structure_hash(c, 2)


def test_training_set_concatenates_earlier_rounds_and_drops_duplicates(tmp_path):
    cfg = Config(home=tmp_path, label={"labeler": "manual"})
    a, b, c = bulk("Al"), bulk("Al", a=4.1), bulk("Al", a=4.2)
    for k, frames in enumerate([[a, b], [b, c], [a]]):
        r = cfg.workdir / f"round_{k:02d}"
        r.mkdir(parents=True)
        data.to_dataframe([labeled(x) for x in frames]).to_pickle(r / "labeled.pkl.gz")
    assert data.training_set(cfg, 0) is None
    assert len(data.training_set(cfg, 1)) == 2
    ts = data.training_set(cfg, 2)      # rounds 0, 1: b twice
    assert len(ts) == 3
    assert [data.structure_hash(x) for x in ts["ase_atoms"]] == \
        [data.structure_hash(x) for x in (a, b, c)]
    assert len(data.training_set(cfg, 3)) == 3


def test_training_set_drops_stress_if_a_round_lacks_it(tmp_path):
    cfg = Config(home=tmp_path, label={"labeler": "manual"})
    for k, stress in enumerate([True, False]):
        r = cfg.workdir / f"round_{k:02d}"
        r.mkdir(parents=True)
        data.to_dataframe([labeled(bulk("Al", a=4 + k / 10), stress=stress)]) \
            .to_pickle(r / "labeled.pkl.gz")
    assert "stress" not in data.training_set(cfg, 2).columns


def test_element_counts():
    df = pd.DataFrame({"ase_atoms": [fcc111("Pd", (3, 3, 4), vacuum=5), bulk("PdO", "rocksalt",
                                                                             a=4.0)]})
    assert data.element_counts(df) == {"O": 1, "Pd": 37}
