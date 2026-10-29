from grace_al.explore.tau_advice import advise

GRID = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5]


def w(tau, collected, rejected=None, stop="max_steps"):
    return {"tau": tau, "collected": collected, "rejected": rejected or {}, "stop_reason": stop}


def test_too_many_rejected_steps_down_the_grid():
    a = advise([w(0.25, 2, {"short_bond": 25, "gamma": 4}), w(0.25, 6, {"gamma": 23})],
               0.2, GRID)
    assert a["verdict"] == "decrease" and a["suggested"] == 0.2
    assert "short_bond" not in a["message"] or "gamma" in a["message"]
    assert "tau too large" in a["message"]
    assert a["per_tau"]["0.25"]["reject_ratio"] == 52 / 60


def test_nothing_collected_steps_up():
    a = advise([w(0.1, 0), w(0.1, 0)], 0.2, GRID)
    assert a["verdict"] == "increase" and a["suggested"] == 0.15


def test_off_grid_scales():
    assert advise([w(0.05, 1, {"gamma": 9})], 0.2, GRID)["suggested"] == 0.03
    assert advise([w(0.7, 0)], 0.2, GRID)["suggested"] == 1.05


def test_largest_good_tau_wins():
    a = advise([w(0.1, 3), w(0.2, 5, {"gamma": 1}), w(0.35, 4, {"gamma": 9})], 0.2, GRID)
    assert a["verdict"] == "ok" and a["suggested"] == 0.2
    assert "0.35 reject too much" in a["message"]


def test_failed_walkers_ignored():
    assert advise([w(0.2, 0, stop="failed")], 0.2, GRID)["verdict"] == "unknown"
