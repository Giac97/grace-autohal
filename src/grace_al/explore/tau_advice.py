"""tau guidance from the walkers of a round: per tau, how much HAL collected and rejected,
and which tau to use next.

A tau is fine when its walkers collect frames and at most explore.max_reject_ratio of the
extrapolative frames are rejected (the calibration's criterion). Too many rejections ->
smaller tau (molecules break up, adsorbates desorb: short_bond, gamma); nothing collected ->
larger tau or higher T. The next value comes from explore.calibrate_taus when it brackets the
current one, else x0.6 / x1.5.
"""

from __future__ import annotations

from collections import Counter


def per_tau(walkers: list[dict]) -> dict[float, dict]:
    """walkers: explore summary rows (tau, collected, rejected {reason: n}, stop_reason)."""
    out: dict[float, dict] = {}
    for w in walkers:
        if w.get("stop_reason") == "failed":
            continue
        r = out.setdefault(float(w["tau"]), {"walkers": 0, "collected": 0, "rejected": 0,
                                             "reasons": Counter()})
        r["walkers"] += 1
        r["collected"] += w.get("collected", 0)
        r["rejected"] += sum((w.get("rejected") or {}).values())
        r["reasons"].update(w.get("rejected") or {})
    for r in out.values():
        n = r["collected"] + r["rejected"]
        r["reject_ratio"] = r["rejected"] / n if n else 0.0
        r["reasons"] = dict(r["reasons"].most_common())
    return out


def _step(tau: float, grid: list[float], down: bool) -> float:
    cands = sorted(t for t in grid if (t < tau if down else t > tau))
    if cands:
        return cands[-1] if down else cands[0]
    return round(tau * (0.6 if down else 1.5), 3)


def advise(walkers: list[dict], max_reject_ratio: float = 0.2,
           grid: list[float] | None = None) -> dict:
    """{"per_tau": {...}, "verdict": ok|decrease|increase, "suggested": tau, "message"}."""
    grid = grid or []
    stats = per_tau(walkers)
    if not stats:
        return {"per_tau": {}, "verdict": "unknown", "suggested": None,
                "message": "no finished walkers"}
    ok = [t for t, r in stats.items()
          if r["collected"] > 0 and r["reject_ratio"] <= max_reject_ratio]
    table = {str(t): r for t, r in sorted(stats.items())}
    if ok:
        best = max(ok)
        too_high = [t for t in stats if t > best]
        msg = f"tau {best:g} collects with {100 * stats[best]['reject_ratio']:.0f} % rejected"
        if too_high:
            msg += f"; tau {', '.join(f'{t:g}' for t in sorted(too_high))} reject too much"
        return {"per_tau": table, "verdict": "ok", "suggested": best, "message": msg}
    if all(r["collected"] == 0 and r["rejected"] == 0 for r in stats.values()):
        t = max(stats)
        new = _step(t, grid, down=False)
        return {"per_tau": table, "verdict": "increase", "suggested": new,
                "message": f"nothing collected at tau {t:g} (tau too small or T too low): "
                           f"try tau {new:g} or a higher T"}
    t = min(stats)
    r = stats[t]
    reason = next(iter(r["reasons"]), "?")
    new = _step(t, grid, down=True)
    return {"per_tau": table, "verdict": "decrease", "suggested": new,
            "message": f"tau {t:g}: {100 * r['reject_ratio']:.0f} % of the extrapolative frames "
                       f"rejected (mostly {reason}), limit {100 * max_reject_ratio:.0f} %: "
                       f"tau too large, try tau {new:g}"}
