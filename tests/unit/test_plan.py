from collections import Counter

import pytest
from ase.build import fcc111

from grace_al.config.schema import Config
from grace_al.explore.plan import (
    PlanError,
    build_plan,
    load_plan,
    save_plan,
    start_order,
    start_pool,
)
from grace_al.systems.base import SystemInfo

SLAB = SystemInfo("slab", ("O", "Pd"), 2)


def frames(names):
    out = []
    for n in names:
        at = fcc111("Pd", (1, 1, 2), vacuum=5)
        at.info["config"] = n
        out.append(at)
    return out


SEEDS = frames(["clean_strain1", "clean_strain1_rattle0", "O_fcc_strain1",
                "O_fcc_strain1_rattle0", "O_hcp_strain1", "Ox2_0", "Ox2_1", "Ox2_0_rattle0"])


def cfg(tmp_path, **explore):
    return Config(home=tmp_path, label={"labeler": "manual"},
                  explore={"temperatures": [800, 1200], "walkers_per_setting": 4, **explore})


def configs_of(plan, starts):
    return [starts[w.start_index].info["config"] for w in plan]


def test_deterministic_per_round(tmp_path):
    a = start_pool(SEEDS, [], 3)
    b = start_pool(SEEDS, [], 3)
    assert [x.info["config"] for x in a] == [x.info["config"] for x in b]
    c = start_pool(SEEDS, [], 4)
    assert [x.info["config"] for x in a] != [x.info["config"] for x in c]
    p1 = build_plan(cfg(tmp_path), 3, a, SLAB)
    p2 = build_plan(cfg(tmp_path), 3, b, SLAB)
    assert p1 == p2


def test_picks_first():
    picks = frames(["a", "b"])
    pool = start_pool(SEEDS, picks, 1)
    assert {x.info["config"] for x in pool[:2]} == {"a", "b"}


def test_clean_fraction_capped_and_distinct_per_setting(tmp_path):
    starts = start_pool(SEEDS, [], 0)
    plan = build_plan(cfg(tmp_path), 0, starts, SLAB)
    assert len(plan) == 8
    names = configs_of(plan, starts)
    assert sum(n.startswith("clean") for n in names) <= 0.2 * len(plan)
    for T in (800, 1200):
        idx = [w.start_index for w in plan if w.temperature == T]
        assert len(set(idx)) == len(idx) == 4


def test_unique_ids_and_seeds(tmp_path):
    plan = build_plan(cfg(tmp_path, tau=[0.1, 0.2]), 2, start_pool(SEEDS, [], 2), SLAB)
    assert len({w.id for w in plan}) == len(plan) == 16
    assert len({w.seed for w in plan}) == 16 and plan[0].seed == 2000
    assert Counter(w.tau for w in plan) == {0.1: 8, 0.2: 8}


def test_only_clean_frames_still_planned():
    starts = frames(["clean_a", "clean_b"])
    assert start_order(starts, 4, {"clean": 0.2}) == [0, 1, 0, 1]


def test_slab_with_npt_is_an_error(tmp_path):
    with pytest.raises(PlanError, match="allow_cell_change_on_slab"):
        build_plan(cfg(tmp_path, ensembles=["nvt", "npt"]), 0, SEEDS, SLAB)
    plan = build_plan(cfg(tmp_path, ensembles=["npt"], allow_cell_change_on_slab=True), 0,
                      SEEDS, SLAB)
    assert {w.ensemble for w in plan} == {"npt"}
    bulk = SystemInfo("bulk", ("Al",))
    assert build_plan(cfg(tmp_path, ensembles=["npt"]), 0, SEEDS, bulk)


def test_cluster_forces_cluster_ensemble(tmp_path):
    plan = build_plan(cfg(tmp_path), 0, SEEDS, SystemInfo("cluster", ("Pd",)))
    assert {w.ensemble for w in plan} == {"cluster"}


def test_save_load(tmp_path):
    plan = build_plan(cfg(tmp_path), 0, SEEDS, SLAB)
    save_plan(tmp_path / "plan.json", plan)
    assert load_plan(tmp_path / "plan.json") == plan
    assert plan[3].dir == "w003"
