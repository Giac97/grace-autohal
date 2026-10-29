"""Model vs DFT: per-element energy alignment, error metrics, parity plots + replot data.

Energies of a fine-tuned model (or the FM) and FHI-aims totals differ by a constant per
element (all-electron references, reference_energy: auto). The model's energies are aligned
by per-element offsets fitted by least squares on the frames compared, so the energy error
is about energetics, not about the zero of energy.

Every plot has a .npz next to it with the plotted arrays; replot with:

    d = np.load("parity.npz"); plt.scatter(d["e_dft"], d["e_pred"])
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from grace_al.report import INK, INK2, SERIES, SURFACE


def compositions(symbols: Sequence[Sequence[str]]) -> tuple[list[str], np.ndarray]:
    """(elements, counts matrix [frames, elements])."""
    els = sorted({s for sym in symbols for s in sym})
    m = np.array([[list(sym).count(e) for e in els] for sym in symbols], dtype=float)
    return els, m


def element_offsets(e_dft, e_pred, symbols) -> dict[str, float]:
    """delta_el minimizing |e_pred - sum n_el delta_el - e_dft| (least squares)."""
    els, m = compositions(symbols)
    if not len(els):
        return {}
    d = np.asarray(e_pred, dtype=float) - np.asarray(e_dft, dtype=float)
    delta = np.linalg.lstsq(m, d, rcond=None)[0]
    return dict(zip(els, map(float, delta), strict=True))


def shift(energies, symbols, offsets: dict[str, float]) -> np.ndarray:
    """energies - sum_el n_el offset_el."""
    return np.array([e - sum(offsets.get(s, 0.0) for s in sym)
                     for e, sym in zip(energies, symbols, strict=True)], dtype=float)


def metrics(ref, pred) -> dict:
    d = np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)
    if not d.size:
        return {"n": 0, "rmse": None, "mae": None, "max": None}
    return {"n": int(d.size), "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mae": float(np.mean(np.abs(d))), "max": float(np.max(np.abs(d)))}


def energy_force_data(symbols, e_dft, e_pred, f_dft, f_pred, groups,
                      offsets: dict[str, float] | None = None) -> dict:
    """Arrays of a parity plot: per-atom energies (model aligned by element offsets) and
    force components, with a group label per frame (strategy, round, ...) and the element of
    every force component."""
    offsets = element_offsets(e_dft, e_pred, symbols) if offsets is None else offsets
    n = np.array([len(s) for s in symbols], dtype=float)
    return {
        "e_dft": np.asarray(e_dft, dtype=float) / n,
        "e_pred": shift(e_pred, symbols, offsets) / n,
        "n_atoms": n.astype(int),
        "group": np.array([str(g) for g in groups]),
        "f_dft": np.concatenate([np.ravel(f) for f in f_dft]) if len(f_dft) else np.zeros(0),
        "f_pred": np.concatenate([np.ravel(f) for f in f_pred]) if len(f_pred) else np.zeros(0),
        "f_element": np.concatenate([np.repeat(np.asarray(s), 3) for s in symbols])
        if len(symbols) else np.zeros(0, dtype=str),
        "offset_elements": np.array(list(offsets)),
        "offset_values": np.array(list(offsets.values()), dtype=float),
    }


# ---------------------------------------------------------------- plots
def _ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#e4e3df")


def _scatter(ax, x, y, labels, title, unit, max_series: int = 3):
    """Parity scatter colored by label (at most max_series colors, the rest 'other'), the
    y = x line, and RMSE / MAE in the corner."""
    x, y, labels = np.asarray(x), np.asarray(y), np.asarray(labels)
    names = [n for n, _ in sorted(((n, -int(np.sum(labels == n))) for n in set(labels)),
                                  key=lambda t: t[1])]
    keep = names[:max_series]
    lab = np.where(np.isin(labels, keep), labels, "other")
    order = keep + (["other"] if len(names) > max_series else [])
    for i, name in enumerate(order):
        m = lab == name
        color = SERIES[i] if name != "other" else "#a3a29b"
        ax.scatter(x[m], y[m], s=14, color=color, alpha=0.75, linewidths=0.4,
                   edgecolors=SURFACE, label=f"{name} ({int(m.sum())})", zorder=2)
    if x.size:
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, linewidth=1,
                zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
    mt = metrics(x, y)
    if mt["n"]:
        ax.text(0.03, 0.97, f"RMSE {mt['rmse']:.3g} {unit}\nMAE  {mt['mae']:.3g} {unit}\n"
                f"n = {mt['n']}", transform=ax.transAxes, va="top", fontsize=8, color=INK,
                family="monospace")
    if order:
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower right")


def plot_energy_forces(d: dict, out_png: Path, title: str) -> Path:
    """Two panels: per-atom energy (by group) and force components (by element)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    _ax_style(a1, "Energy (element offsets removed)", "DFT (eV/atom)", "model (eV/atom)")
    _scatter(a1, d["e_dft"], d["e_pred"], d["group"], "energy", "eV/atom")
    _ax_style(a2, "Force components", "DFT (eV/Å)", "model (eV/Å)")
    _scatter(a2, d["f_dft"], d["f_pred"], d["f_element"], "forces", "eV/Å")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def plot_property(x, y, labels, out_png: Path, title: str, xlabel: str, unit: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    _ax_style(ax, title, f"DFT {xlabel} ({unit})", f"model {xlabel} ({unit})")
    _scatter(ax, x, y, labels, title, unit)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def save(out_png: Path, d: dict) -> Path:
    """The plotted arrays next to the image (same name, .npz)."""
    out = out_png.with_suffix(".npz")
    np.savez(out, **d)
    return out
