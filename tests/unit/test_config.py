import shutil
from pathlib import Path

import pytest
import yaml

from grace_al.config.loader import ConfigError, dump_config, env_overrides, load_config
from grace_al.config.schema import DEFAULT_AIMS_PARAMETERS, Config

DATA = Path(__file__).parent / "data"
TEMPLATES = Path(__file__).parents[2] / "src/grace_al/config/templates"


def write(tmp_path, d, name="config.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d))
    return p


def minimal(**kw):
    d = {"schema_version": 1, "label": {"labeler": "teacher",
                                        "teacher_model": "GRACE-2L-OMAT-large-ft-E"}}
    d.update(kw)
    return d


# ---------------------------------------------------------------- legacy config
def test_legacy_config_maps_old_keys(tmp_path):
    p = tmp_path / "config.yaml"
    shutil.copy(DATA / "legacy_config.yaml", p)
    cfg = load_config(p)
    assert cfg.explore.backend == "pool"             # parallel_cli: true
    assert cfg.explore.walkers_per_setting == 4      # repeats: 4
    assert cfg.explore.temperatures == [800, 1200]
    assert cfg.label.labeler == "aims"
    assert cfg.convergence.max_rounds == 5
    assert cfg.system.adsorbate.species == "CO"
    assert cfg.aims.parameters["elsi_build_dm_method"] == "pdgemm"
    assert cfg.aims.command == tmp_path / "run_aims.sh"
    assert cfg.seeds == tmp_path / "seeds.extxyz"


def test_legacy_defaults_keep_old_behavior(tmp_path):
    cfg = load_config(write(tmp_path, {"hal": {"ensembles": ["nvt"]}}))
    assert cfg.explore.backend == "inproc"           # old default: no parallel_cli
    assert cfg.label.labeler == "manual"             # old default: labeler null


def test_legacy_fm_labeler(tmp_path):
    cfg = load_config(write(tmp_path, {"labeler": "fm:GRACE-2L-OMAT-large-ft-E"}))
    assert (cfg.label.labeler, cfg.label.teacher_model) == ("teacher",
                                                            "GRACE-2L-OMAT-large-ft-E")


@pytest.mark.parametrize("src", [DATA / "legacy_config.yaml", TEMPLATES / "surface.yaml",
                                 TEMPLATES / "custom_seeds.yaml", TEMPLATES / "bulk.yaml"])
def test_round_trip(tmp_path, src):
    p = tmp_path / "config.yaml"
    shutil.copy(src, p)
    cfg = load_config(p)
    dump_config(cfg, tmp_path / "again.yaml")
    again = load_config(tmp_path / "again.yaml")
    assert again == cfg
    assert yaml.safe_load((tmp_path / "again.yaml").read_text())["seeds"] == "seeds.extxyz"


# ---------------------------------------------------------------- validation
def test_unknown_key_names_its_path(tmp_path):
    d = minimal(aims={"species_dir": "x", "paramters": {"xc": "pbe"}})
    with pytest.raises(ConfigError, match=r"aims\.paramters: Extra inputs are not permitted"):
        load_config(write(tmp_path, d))


@pytest.mark.parametrize("override, match", [
    ({"foundation_model": "GRACE-2L-OMAT-L"}, "alias; use the full name"),
    ({"explore": {"tau": []}}, r"explore\.tau"),
    ({"explore": {"temperatures": []}}, r"explore\.temperatures"),
    ({"aims": {"species_dir": "x", "kspacing": 0}}, r"aims\.kspacing: .*greater than 0"),
    ({"label": {"labeler": "aims"}}, "no `aims:` section"),
    ({"label": {"labeler": "aims"}, "aims": {}}, "set aims.species_dir"),
    ({"label": {"labeler": "teacher"}}, "needs `teacher_model`"),
    ({"label": {"labeler": "vasp"}}, "labeler 'vasp'"),
    ({"label": {"labeler": "teacher", "teacher_model": "m", "mode": "array"}},
     "needs an `hpc:` section"),
    ({"explore": {"ensembles": ["nve"]}}, r"explore\.ensembles\.0"),
    ({"system": {"generator": "surface"}}, "needs a `surface:` section"),
])
def test_validators(tmp_path, override, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, minimal(**override)))


def test_aims_parameters_on_top_of_defaults(tmp_path):
    d = minimal(aims={"species_dir": "sp", "parameters": {"xc": "pw-lda",
                                                          "use_dipole_correction": None}})
    p = load_config(write(tmp_path, d)).aims.parameters
    assert p["xc"] == "pw-lda"
    assert "use_dipole_correction" not in p
    assert p["sc_iter_limit"] == DEFAULT_AIMS_PARAMETERS["sc_iter_limit"]


# ---------------------------------------------------------------- paths
def test_relative_paths_resolve_against_config_folder(tmp_path, monkeypatch):
    sub = tmp_path / "campaign"
    sub.mkdir()
    monkeypatch.chdir(tmp_path)
    d = minimal(workdir="out", seeds="data/s.extxyz",
                aims={"species_dir": "../species", "command": "run_aims.sh"})
    cfg = load_config(write(sub, d))
    assert cfg.home == sub
    assert cfg.workdir == sub / "out"
    assert cfg.seeds == sub / "data/s.extxyz"
    assert cfg.aims.species_dir.resolve() == tmp_path / "species"
    assert cfg.aims.command == sub / "run_aims.sh"


def test_absolute_paths_stay(tmp_path):
    cfg = load_config(write(tmp_path, minimal(aims={"species_dir": "/opt/species"})))
    assert cfg.aims.species_dir == Path("/opt/species")
    assert cfg.to_dict()["aims"]["species_dir"] == "/opt/species"


# ---------------------------------------------------------------- layers
def test_site_profile_then_user_then_env(tmp_path):
    d = minimal(site="bwunicluster3", aims={"species_dir": "sp", "np": 16},
                hpc={"partition_gpu": "gpu_mine"})
    env = {"GRACE_AL__AIMS__NP": "48", "GRACE_AL__HPC__ENV__I_MPI_DEBUG": "5"}
    cfg = load_config(write(tmp_path, d), environ=env)
    assert cfg.aims.parameters["elsi_build_dm_method"] == "pdgemm"   # site
    assert cfg.aims.parameters["xc"] == "pbe"                        # schema default
    assert cfg.hpc.partition_gpu == "gpu_mine"                       # user over site
    assert cfg.hpc.modules[1] == "mpi/impi/2021.14-intel-2025.0"     # site
    assert cfg.hpc.env["I_MPI_COLL_INTRANODE"] == "pt2pt"            # site
    assert cfg.hpc.env["I_MPI_DEBUG"] == "5"                         # env
    assert cfg.aims.np == 48                                         # env over user


def test_cli_site_wins_over_config_site(tmp_path):
    d = minimal(site="bwunicluster3")
    cfg = load_config(write(tmp_path, d), site="workstation")
    assert cfg.site == "workstation"
    assert cfg.hpc.mpirun == "/usr/lib64/openmpi/bin/mpirun"
    assert cfg.hpc.modules == []


def test_unknown_site(tmp_path):
    with pytest.raises(ConfigError, match="unknown site 'nowhere'"):
        load_config(write(tmp_path, minimal(site="nowhere")))


def test_env_overrides_parse_yaml_scalars():
    env = {"GRACE_AL__EXPLORE__TAU": "[0.1, 0.2]", "GRACE_AL__FINETUNE__FROZEN": "false",
           "OTHER": "1"}
    assert env_overrides(env) == {"explore": {"tau": [0.1, 0.2]},
                                  "finetune": {"frozen": False}}


# ---------------------------------------------------------------- fingerprints
def test_fingerprint_stable_under_key_order(tmp_path):
    a = {"xc": "pbe", "sc_iter_limit": 100}
    b = dict(reversed(list(a.items())))
    fa = load_config(write(tmp_path, minimal(aims={"species_dir": "s", "parameters": a}),
                           "a.yaml")).section_fingerprint("aims", "label")
    fb = load_config(write(tmp_path, minimal(aims={"species_dir": "s", "parameters": b}),
                           "b.yaml")).section_fingerprint("label", "aims")
    assert fa == fb


def test_fingerprint_changes_with_aims_parameter(tmp_path):
    f = lambda v: load_config(write(tmp_path, minimal(
        aims={"species_dir": "s", "parameters": {"sc_accuracy_rho": v}}))
    ).section_fingerprint("aims")
    assert f(1e-6) != f(1e-5)


def test_fingerprint_independent_of_home(tmp_path):
    d = minimal(aims={"species_dir": "species", "command": "run_aims.sh"})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    fa = load_config(write(tmp_path / "a", d)).section_fingerprint("aims", "explore")
    fb = load_config(write(tmp_path / "b", d)).section_fingerprint("aims", "explore")
    assert fa == fb


def test_fingerprint_unknown_section():
    with pytest.raises(KeyError):
        Config(label={"labeler": "manual"}).section_fingerprint("nope")
