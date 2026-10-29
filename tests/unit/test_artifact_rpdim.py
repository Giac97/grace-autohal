import pytest

from grace_al import _tp
from grace_al.config.schema import Config
from grace_al.stages.artifact import TooFewAtoms, build_params, capped_rp_dim


@pytest.mark.parametrize("rp_max, counts, expected", [
    (32, {"Pd": 1000, "O": 12}, 8),        # 12 atoms -> 8
    (32, {"Pd": 1000, "O": 100}, 32),      # enough atoms: rp_max
    (128, {"Pd": 50, "O": 40}, 36),
])
def test_cap(rp_max, counts, expected):
    assert capped_rp_dim(rp_max, counts) == expected


def test_too_few_atoms():
    with pytest.raises(TooFewAtoms, match="add seeds"):
        capped_rp_dim(32, {"Pd": 1000, "O": 7})   # 7 atoms -> 3 < 8


def test_round_params(tmp_path):
    cfg = Config(home=tmp_path, label={"labeler": "manual"},
                 artifact={"rp_dim_round0": 32, "rp_dim": 64, "n_clusters": [1, 4]})
    assert build_params(cfg, 0, {"Pd": 500, "O": 500}) == \
        {"rp_dim": 32, "rp_max": 32, "n_clusters": [1]}
    assert build_params(cfg, 1, {"Pd": 500, "O": 500}) == \
        {"rp_dim": 64, "rp_max": 64, "n_clusters": [1, 4]}


SAMPLE_INFO = """
  Element  K_eff   n_atoms    n/clust    max_cond  min_rank  max_trunc
  ----------------------------------------------------------------------
     O(0)      1        24      24..24    3.10e+04      8/8          0
    Pd(1)      1      1350  1350..1350    1.20e+03    32/32          0 (3 sentinel)
"""


def test_parse_uq_info():
    assert _tp.parse_uq_info(SAMPLE_INFO) == {
        "O": {"K_eff": 1, "n_atoms": 24, "min_rank": 8},
        "Pd": {"K_eff": 1, "n_atoms": 1350, "min_rank": 32}}


def test_parse_select_log():
    natural = "  calm-greedy: 7 frame(s) picked, gamma at pick 3.1 -> 1.2\n" \
              "  natural set size: 7 (every remaining frame has max gamma <= 1 under ...)\n"
    capped = "  -n 100 reached before the natural set: remaining max gamma 2.1 > 1\n"
    assert _tp.parse_select_log(natural) == {"natural": True, "natural_set": 7}
    assert _tp.parse_select_log(capped)["natural"] is False


def test_fit_reference_energies(tmp_path):
    assert _tp.fit_reference_energies(tmp_path) is None
    (tmp_path / "log.txt").write_text(
        "INFO reference_energy=auto: computed single-atom energies (eV): {'O': -2.0}\n"
        "INFO reference_energy=auto: computed single-atom energies (eV): "
        "{'O': -2047.1, 'Pd': -136000.5}\n")
    assert _tp.fit_reference_energies(tmp_path) == {"O": -2047.1, "Pd": -136000.5}


def test_fit_reference_energies_in_seed_folder(tmp_path):
    """gracemaker writes log.txt into fit/seed/1/ (real layout)."""
    (tmp_path / "seed" / "1").mkdir(parents=True)
    (tmp_path / "seed" / "1" / "log.txt").write_text(
        "2026-10-29 14:45:16,956 I - reference_energy=auto: computed single-atom energies "
        "(eV): {'O': -6.123339195302772, 'Pd': -4.973367337893022}\n")
    assert _tp.fit_reference_energies(tmp_path) == {"O": -6.123339195302772,
                                                    "Pd": -4.973367337893022}
