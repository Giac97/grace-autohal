import numpy as np
import pytest
from ase.build import add_adsorbate, bulk, fcc111

from grace_al import properties as P


def slab(strain=1.0, ads=(), seed=0):
    s = fcc111("Pd", (2, 2, 3), vacuum=8, periodic=True)
    s.set_cell(s.cell.array * [[strain], [strain], [1]], scale_atoms=True)
    for el, site in ads:
        add_adsorbate(s, el, 1.2, site)
    s.rattle(0.01, seed=seed)
    return s


def test_relative_within_composition_and_cell():
    frames = [slab(ads=[("O", "fcc")]), slab(ads=[("O", "hcp")]), slab(ads=[("O", "ontop")]),
              slab(1.02, ads=[("O", "fcc")]), slab()]
    e_dft = np.array([-10.0, -9.8, -9.0, -9.5, -5.0])
    e_pred = e_dft + np.array([3.0, 3.1, 2.9, 7.0, 0.0])        # any constant per group is fine
    idx, x, y = P.relative(frames, e_dft, e_pred)
    assert list(idx) == [1, 2]                                    # strained / clean: alone
    assert x == pytest.approx([0.2, 1.0]) and y == pytest.approx([0.3, 0.9])


def test_adsorption_with_references():
    clean, clean2 = slab(), slab(1.02)
    frames = [clean, clean2, slab(ads=[("O", "fcc")]), slab(ads=[("O", "fcc"), ("O", "hcp")]),
              slab(1.02, ads=[("O", "ontop")])]
    e_dft = np.array([-100.0, -99.0, -106.0, -111.0, -104.5])
    idx, x, y, names, how = P.adsorption(frames, e_dft, e_dft + 1.0, {"Pd"},
                                         {"O": -5.0}, {"O": -5.0})
    assert how == "references" and list(idx) == [2, 3, 4]
    assert x == pytest.approx([-1.0, -1.0, -0.5])
    assert names == ["O", "O×2", "O"]
    assert y == pytest.approx(x)          # +1 eV on every structure cancels against the slab


def test_adsorption_fitted_mu_keeps_differences():
    frames = [slab(), slab(ads=[("O", "fcc")]), slab(ads=[("O", "hcp")])]
    e_dft = np.array([-100.0, -106.0, -105.8])
    _, x, _, _, how = P.adsorption(frames, e_dft, e_dft, {"Pd"})
    assert how == "fitted" and x[1] - x[0] == pytest.approx(0.2)


def test_cohesive_needs_atoms():
    al = bulk("Al", cubic=True)
    frames = [al, slab()]
    e = np.array([-4 * 3.4, -50.0])
    idx, x, _ = P.cohesive(frames, e, e, {"Al": 0.0}, {"Al": 0.0})
    assert list(idx) == [0] and x == pytest.approx([3.4])       # the slab is skipped
    with pytest.raises(P.PropertyError, match="isolated-atom"):
        P.cohesive([bulk("Cu")], np.array([-3.0]), np.array([-3.0]), {}, {})


def test_fit_mu_from_references():
    from ase import Atoms

    refs = [Atoms("O"), Atoms("CO"), Atoms("C")]
    mu = P.fit_mu(refs, [-2.0, -7.0, -5.0])
    assert mu == pytest.approx({"C": -5.0, "O": -2.0})


def test_adsorbate_label():
    assert P.adsorbate_label({"C": 2, "O": 2}) == "CO×2"
    assert P.adsorbate_label({"O": 1, "H": 1}) == "HO"


def test_resolve_model_errors(fake_runs):
    from grace_al.config.loader import load_config

    cfg = load_config(fake_runs)
    with pytest.raises(P.PropertyError, match="artifact stage"):
        P.resolve_model(cfg, "fm")
    with pytest.raises(P.PropertyError, match="UQ artifact"):
        P.resolve_model(cfg, "last")
    with pytest.raises(P.PropertyError, match="no finished fit"):
        P.resolve_model(cfg, "round:7")


def test_validate_cli_end_to_end(tmp_path, monkeypatch):
    """`grace-al validate` with predictions faked (no TF): plots + npz for each property."""
    import yaml
    from ase.calculators.singlepoint import SinglePointCalculator
    from typer.testing import CliRunner

    from grace_al import data
    from grace_al.cli import app
    from grace_al.models import ModelRef

    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "schema_version": 1, "label": {"labeler": "manual"},
        "system": {"generator": "surface", "surface": {"element": "Pd"}}}))
    frames = [slab(), slab(ads=[("O", "fcc")]), slab(ads=[("O", "hcp")], seed=1),
              slab(ads=[("O", "ontop")], seed=2)]
    rng = np.random.default_rng(0)
    for i, (at, e) in enumerate(zip(frames, [-100.0, -106.0, -105.8, -105.0], strict=True)):
        at.calc = SinglePointCalculator(at, energy=e, forces=rng.normal(size=(len(at), 3)))
        r = tmp_path / "runs" / f"round_{i // 2:02d}"
        r.mkdir(parents=True, exist_ok=True)
    data.to_dataframe(frames[:2]).to_pickle(tmp_path / "runs/round_00/labeled.pkl.gz")
    data.to_dataframe(frames[2:]).to_pickle(tmp_path / "runs/round_01/labeled.pkl.gz")
    monkeypatch.setattr(P, "resolve_model", lambda cfg, spec: (
        ModelRef("finetuned", tmp_path / "m", tmp_path / "c", tmp_path / "a"), 0, "round00"))
    truth = {data.structure_hash(at): (at.get_potential_energy(), at.get_forces())
             for at in frames}
    monkeypatch.setattr(P, "predict", lambda model, fr, work: (
        np.array([truth[data.structure_hash(at)][0] + 50.0 for at in fr]),
        [truth[data.structure_hash(at)][1] + 0.01 for at in fr]))
    r = CliRunner().invoke(app, ["validate", str(p)])
    assert r.exit_code == 0, r.output
    out = tmp_path / "runs" / "report" / "validate"
    for prop in ("energy", "relative", "adsorption"):
        assert (out / f"{prop}_round00.png").exists() and (out / f"{prop}_round00.npz").exists()
    d = np.load(out / "relative_round00.npz")
    assert d["dft"] == pytest.approx(d["model"]) and set(d["label"]) == {"new"}
    assert "forces RMSE 0.01 eV/Å" in r.output
