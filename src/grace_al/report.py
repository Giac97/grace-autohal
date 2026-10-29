"""Round history of a campaign (`grace-al status`, `grace-al report`).

Everything is read from files the stages wrote - done-marker summaries, validation.json,
failures.json, fit/seed/1/test_metrics.yaml, selected.extxyz - so it works on a copy of
runs/ without TensorFlow. Plots go to runs/report/*.png; --html adds a self-contained
runs/report/index.html (plots embedded, plus the numbers as a table).

GPU idle: in inline mode the GPU waits while aims runs in the same job, so the label
stage's wall time over the round's wall time (from the markers) is the GPU idle fraction.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from grace_al.state import STAGES, read_marker, round_dir, rounds_on_disk

if TYPE_CHECKING:
    from grace_al.config.schema import Config

# reference palette (dataviz skill), light mode; text never in series colors
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def _seconds(m: dict | None) -> float | None:
    if not m or not m.get("started") or not m.get("finished"):
        return None
    t = lambda s: datetime.fromisoformat(s)
    return (t(m["finished"]) - t(m["started"])).total_seconds()


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def fit_test_metrics(fit: Path) -> dict | None:
    """Last epoch of gracemaker's test metrics: force and energy RMSE."""
    import yaml

    p = fit / "seed" / "1" / "test_metrics.yaml"
    if not p.exists():
        return None
    rows = yaml.safe_load(p.read_text()) or []
    if not rows:
        return None
    last = rows[-1]
    return {"force_rmse": last.get("rmse/f_comp"), "energy_rmse_per_atom": last.get("rmse/depa"),
            "epoch": last.get("epoch")}


def selected_gammas(rdir: Path) -> list[float]:
    """Max gamma (pristine) of each selected frame, as HAL recorded it."""
    p = rdir / "selected.extxyz"
    if not p.exists() or not p.stat().st_size:
        return []
    from grace_al.data import read_frames

    out = []
    for at in read_frames(p):
        g = at.info.get("gamma_max_pristine", at.info.get("gamma_max_live"))
        if g is not None:
            out.append(float(g))
    return out


def round_rows(cfg: Config, gammas: bool = False) -> list[dict]:
    """One dict per round directory: what each stage reported."""
    rows = []
    for k in rounds_on_disk(cfg.workdir):
        rdir = round_dir(cfg.workdir, k)
        m = {s: read_marker(rdir, s) for s in STAGES}
        sm = {s: (v or {}).get("summary", {}) for s, v in m.items()}
        val = _json(rdir / "validation.json")
        fails = _json(rdir / "failures.json")
        times = {s: _seconds(v) for s, v in m.items()}
        total = sum(t for t in times.values() if t)
        row = {
            "round": k,
            "done": [s for s in STAGES if m[s]],
            "walkers": sm["explore"].get("n_walkers"),
            "collected": sm["explore"].get("collected"),
            "rejected": sum((sm["explore"].get("rejected") or {}).values())
            if m["explore"] else None,
            "selected": sm["select"].get("n_selected"),
            "natural_set": sm["select"].get("natural_set"),
            "picks": sm["select"].get("picks"),
            "labeled": sm["label"].get("n_labeled"),
            "failed": sm["label"].get("n_failed"),
            "failures": (fails or {}).get("counts") or sm["label"].get("failures") or {},
            "aims_version": (m["label"] or {}).get("versions", {}).get("aims"),
            "fit_done": (rdir / "fit" / "FINISHED").exists(),
            "fit_test": fit_test_metrics(rdir / "fit"),
            "val_force_rmse": val.get("force_rmse") if val else None,
            "val_energy_rmse": val.get("energy_rmse_per_atom") if val else None,
            "val_by_strategy": {s: v.get("force_rmse")
                                for s, v in (val or {}).get("by_strategy", {}).items()},
            "converged": next((sm[s].get("converged") for s in ("select", "validate")
                               if sm[s].get("converged")), None),
            "seconds": total or None,
            "label_seconds": times["label"],
            "label_mode": sm["label"].get("mode"),
        }
        if gammas:
            row["gammas"] = selected_gammas(rdir)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- plots
def _axes(title: str, ylabel: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    return fig, ax


def _rounds_axis(ax, rounds):
    ax.set_xticks(rounds)
    ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
    ax.set_xlabel("round", fontsize=8.5, color=INK2)


def _save(fig, out: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def _legend(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)


def _bars(ax, x, series: list[tuple[str, list]], stacked: bool = False):
    """Grouped (or stacked) bars with a 2 px surface gap, colors in slot order."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    n = len(series)
    width = 0.8 / (1 if stacked else n)
    bottom = np.zeros(len(x))
    for i, (label, vals) in enumerate(series):
        v = np.array([0 if y is None else y for y in vals], dtype=float)
        pos = x if stacked else x - 0.4 + width * (i + 0.5)
        ax.bar(pos, v, width, bottom=bottom if stacked else None, label=label,
               color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=1.5)
        if stacked:
            bottom += v


def _line(ax, x, y, label, i, dodge: float = 0.0):
    """Series i, shifted by `dodge` rounds so equal values of several series stay visible;
    magnitudes, so the axis starts at 0."""
    pts = [(a + dodge, b) for a, b in zip(x, y, strict=True) if b is not None]
    if pts:
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=2 + i)
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.15 * max(ys)))
    else:
        ax.set_ylim(bottom=0)


def make_plots(rows: list[dict], out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    x = [r["round"] for r in rows]
    plots: dict[str, Path] = {}
    if not rows:
        return plots

    fig, ax = _axes("HAL frames per round", "frames")
    _bars(ax, x, [("collected", [r["collected"] for r in rows]),
                  ("rejected", [r["rejected"] for r in rows])])
    _rounds_axis(ax, x)
    _legend(ax)
    plots["exploration"] = _save(fig, out / "exploration.png")

    fig, ax = _axes("Natural set size (calm-greedy)", "frames")
    _line(ax, x, [r["natural_set"] for r in rows], "natural set", 0)
    _rounds_axis(ax, x)
    plots["natural_set"] = _save(fig, out / "natural_set.png")

    fig, ax = _axes("Validation: previous model on the new DFT data", "force RMSE (eV/Å)")
    _line(ax, x, [r["val_force_rmse"] for r in rows], "all", 0, -0.08)
    for i, s in enumerate(("calm-greedy", "random-all"), start=1):
        _line(ax, x, [r["val_by_strategy"].get(s) for r in rows], s, i, 0.08 * (i - 1))
    _rounds_axis(ax, x)
    _legend(ax)
    plots["validation"] = _save(fig, out / "validation.png")

    fig, ax = _axes("Fine-tune test set", "force RMSE (eV/Å)")
    _line(ax, x, [(r["fit_test"] or {}).get("force_rmse") for r in rows], "test", 0)
    _rounds_axis(ax, x)
    plots["fit"] = _save(fig, out / "fit.png")

    if any(r.get("gammas") for r in rows):
        fig, ax = _axes("Max γ of the selected frames", "max γ (pristine)")
        data = [r.get("gammas") or [float("nan")] for r in rows]
        parts = ax.boxplot(data, positions=x, widths=0.5, patch_artist=True,
                           medianprops={"color": INK, "linewidth": 1.5},
                           whiskerprops={"color": INK2}, capprops={"color": INK2},
                           flierprops={"markeredgecolor": INK2, "markersize": 4})
        for b in parts["boxes"]:
            b.set(facecolor=SERIES[0], edgecolor=SURFACE)
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
        ax.annotate("γ = 1 (known)", (0, 1.0), xycoords=("axes fraction", "data"),
                    xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        _rounds_axis(ax, x)
        plots["gamma"] = _save(fig, out / "gamma.png")

    cats = sorted({c for r in rows for c in r["failures"]})
    if cats:
        fig, ax = _axes("Labeling failures by category", "calculations")
        _bars(ax, x, [(c, [r["failures"].get(c, 0) for r in rows]) for c in cats],
              stacked=True)
        _rounds_axis(ax, x)
        _legend(ax)
        plots["failures"] = _save(fig, out / "failures.png")

    if any(r["seconds"] for r in rows):
        fig, ax = _axes("Wall time per round (GPU idle while DFT runs inline)", "hours")
        lab = [(r["label_seconds"] or 0) / 3600 if r["label_mode"] != "array" else 0
               for r in rows]
        rest = [((r["seconds"] or 0) / 3600) - a for r, a in zip(rows, lab, strict=True)]
        _bars(ax, x, [("GPU stages", rest), ("DFT in the GPU job", lab)], stacked=True)
        _rounds_axis(ax, x)
        _legend(ax)
        plots["time"] = _save(fig, out / "time.png")
    return plots


# ---------------------------------------------------------------- html
COLUMNS = [("round", "round"), ("walkers", "walkers"), ("collected", "collected"),
           ("rejected", "rejected"), ("selected", "selected"), ("natural_set", "natural set"),
           ("labeled", "labeled"), ("failed", "failed"), ("val_force_rmse", "val. F RMSE"),
           ("fit_done", "fit"), ("converged", "converged")]


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def write_html(rows: list[dict], plots: dict[str, Path], out: Path, title: str) -> Path:
    imgs = "".join(
        f'<figure><img alt="{name}" src="data:image/png;base64,'
        f'{base64.b64encode(p.read_bytes()).decode()}"></figure>' for name, p in plots.items())
    head = "".join(f"<th>{h}</th>" for _, h in COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r[c]))}</td>" for c, _ in COLUMNS)
                   + "</tr>" for r in rows)
    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --grid:{GRID}; }}
body {{ margin:0; padding:24px 16px; background:var(--surface); color:var(--ink);
  font:14px/1.45 system-ui, sans-serif; }}
main {{ max-width:1120px; margin:auto; }}
h1 {{ font-size:20px; margin:0 0 4px; }} p {{ color:var(--ink2); margin:0 0 20px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }}
figure {{ margin:0; }} img {{ width:100%; height:auto; display:block; }}
.table {{ overflow-x:auto; margin-top:24px; }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ padding:4px 10px; text-align:right; border-bottom:1px solid var(--grid); }}
th {{ color:var(--ink2); font-weight:500; }}
</style></head><body><main><h1>{html.escape(title)}</h1>
<p>{len(rows)} round(s). Generated by grace-al report.</p>
<div class="grid">{imgs}</div>
<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
</main></body></html>
""")
    return out


def write_report(cfg: Config, html_page: bool = False) -> tuple[Path, list[dict]]:
    out = cfg.workdir / "report"
    rows = round_rows(cfg, gammas=True)
    plots = make_plots(rows, out)
    (out / "rounds.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    if html_page:
        write_html(rows, plots, out / "index.html", f"grace-al: {cfg.home.name}")
    return out, rows
