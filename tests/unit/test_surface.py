import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk as ase_bulk
from ase.io import write

from grace_al.config.loader import load_config
from grace_al.config.schema import Config
from grace_al.data import structure_hash
from grace_al.stages.seeds import summary
from grace_al.systems.base import get_generator
from grace_al.systems.bulk import BulkGenerator
from grace_al.systems.from_file import FileGenerator
from grace_al.systems.surface import SurfaceGenerator, configs, make_slab, strain_lateral

DATA = Path(__file__).parent / "data"


def template_cfg(tmp_path, **system_initial):
    d = yaml.safe_load((resources.files("grace_al.config.templates") / "surface.yaml")
                       .read_text())
    d["system"]["initial"].update({"fm_relax": False, **system_initial})
    d["label"] = {"labeler": "manual"}
    d.pop("aims", None)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(d))
    return load_config(tmp_path / "config.yaml")


def test_default_config_same_as_build_seeds(tmp_path):
    """CO/Pd(111) template, no relaxation: identical to the original build_seeds.py (structure
    hashes recorded from it), 19 configs x 2."""
    frames = SurfaceGenerator().build(template_cfg(tmp_path))
    ref = json.loads((DATA / "seeds_co_pd111_unrelaxed.json").read_text())
    assert len(frames) == 38
    assert [at.info["config"] for at in frames] == list(ref)
    assert {at.info["config"]: structure_hash(at, 6) for at in frames} == ref
    assert all(at.pbc.all() for at in frames)
    assert all(at.info["generator"] == "surface" for at in frames)


def test_labels(tmp_path):
    names = [at.info["config"] for at in configs(template_cfg(tmp_path))]
    assert names[:3] == ["clean_strain0.98", "clean_strain1", "clean_strain1.02"]
    assert "CO_fcc_strain1" in names and "COx3_1" in names and len(names) == 19


def test_facets_and_lateral_strain(tmp_path):
    cfg = template_cfg(tmp_path)
    for facet in ("fcc100", "fcc111", "fcc211"):
        s = cfg.system.surface.model_copy(update={"facet": facet})
        assert make_slab(s).pbc.all()
    slab = make_slab(cfg.system.surface)
    st = strain_lateral(slab, 1.02)
    assert np.allclose(st.cell[:2], slab.cell[:2] * 1.02)
    assert np.allclose(st.cell[2], slab.cell[2])
    assert np.allclose(st.positions[:, 2], slab.positions[:, 2])


def test_min_distance_respected(tmp_path):
    from grace_al.systems.adsorbates import lateral_distance

    cfg = template_cfg(tmp_path, coverages=[3], n_per_coverage=4)
    for at in configs(cfg):
        if "x3" not in at.info["config"]:
            continue
        c = [a.position[:2] for a in at if a.symbol == "C"]
        assert len(c) == 3
        for i in range(3):
            for j in range(i):
                assert lateral_distance(c[i], c[j], at.cell) >= 2.5 - 1e-6


def test_several_species_and_mix(tmp_path):
    cfg = template_cfg(tmp_path)
    d = cfg.to_dict()
    d["system"]["adsorbate"] = [{"species": "CO"}, {"species": "O", "sites": ["fcc", "hcp"]}]
    d["system"]["initial"].update({"strains": [1.0], "coverages": [2], "mix": True,
                                   "n_per_coverage": 3})
    (tmp_path / "c2.yaml").write_text(yaml.safe_dump(d))
    cfg2 = load_config(tmp_path / "c2.yaml")
    names = [at.info["config"] for at in configs(cfg2)]
    assert "O_fcc_strain1" in names and "O_ontop_strain1" not in names
    mixed = [n for n in names if "x2_" in n]
    assert len(mixed) == 3 and all(n.split("x2")[0] in ("CO", "O", "CO+O") for n in mixed)
    assert SurfaceGenerator().info(cfg2).elements == ("C", "O", "Pd")


def test_info(tmp_path):
    i = SurfaceGenerator().info(template_cfg(tmp_path))
    assert (i.kind, i.elements, i.vacuum_axis) == ("slab", ("C", "O", "Pd"), 2)
    assert type(get_generator("surface")).__name__ == "SurfaceGenerator"


def test_seed_summary_warns_on_scarce_elements(tmp_path):
    s = summary(SurfaceGenerator().build(template_cfg(tmp_path)), 32, "surface")
    assert s["n_seeds"] == 38 and s["elements"]["C"] == 2 * (12 + 2 * 2 + 2 * 3)
    assert s["scarce"] == {} and "warning" not in s              # 44 >= 32 + 4
    s = summary(SurfaceGenerator().build(template_cfg(tmp_path)), 64, "surface")
    assert set(s["scarce"]) == {"C", "O"} and "n_rattle" in s["warning"]


# ---------------------------------------------------------------- bulk, file
def test_bulk_generator(tmp_path):
    cfg = Config(home=tmp_path, label={"labeler": "manual"},
                 system={"generator": "bulk",
                         "bulk": {"element": "Al", "size": [2, 2, 2], "vacancies": [0, 1],
                                  "substitutions": {"Cu": 0.125}},
                         "initial": {"strains": [1.0], "n_per_coverage": 2, "n_rattle": 1,
                                     "fm_relax": False}})
    frames = BulkGenerator().build(cfg)
    assert len(frames) == 2 * 2 * 2          # (vac0, vac1) x 2 variants, + rattled
    sym = [at.get_chemical_symbols() for at in frames]
    assert all(s.count("Cu") == 4 for s in sym)
    assert sorted({len(at) for at in frames}) == [31, 32]
    assert BulkGenerator().info(cfg).elements == ("Al", "Cu")
    assert BulkGenerator().info(cfg).kind == "bulk"


def test_file_generator(tmp_path):
    write(tmp_path / "mine.extxyz", [ase_bulk("Al", cubic=True)] * 2)
    cfg = Config(home=tmp_path, label={"labeler": "manual"},
                 system={"generator": "file", "path": "mine.extxyz"})
    frames = FileGenerator().build(cfg)
    assert len(frames) == 2 and frames[0].info["generator"] == "file"
    assert FileGenerator().info(cfg).kind == "bulk"
    bad = ase_bulk("Al", cubic=True)
    bad.pbc = [True, True, False]
    write(tmp_path / "mine.extxyz", [bad])
    with pytest.raises(ValueError, match="pbc"):
        FileGenerator().build(cfg)
