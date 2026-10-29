"""Property parity (`grace-al validate`): derived energies of a model vs DFT.

The model (foundation model, or the one fine-tuned in round K) is evaluated on labeled data
(all rounds by default) in one `grace_uq predict` subprocess; every property is then computed
the same way from the DFT and from the model energies:

  energy      per-atom energy (model aligned by per-element offsets) + force components
  relative    E - E(DFT-lowest frame) within frames of one composition and cell
              (site preference, coverage ordering; no references needed)
  adsorption  E(slab + adsorbates) - E(clean slab, same cell) - sum_ads n_el mu_el
  cohesive    sum_el x_el mu_el(isolated atom) - E/N of frames without vacuum

mu_el comes from --references: a structure file with DFT-labeled references (isolated atoms,
gas molecules); the model computes the same structures itself, so each side uses its own
references. Without references, adsorption uses mu_el fitted by least squares on the data
(per method): absolute adsorption energies are then only defined up to a constant per
adsorbate element, but differences between sites and coverages are kept. Cohesive energies
need isolated-atom references.

Frames are labeled "trained on" (labeled in rounds <= K, the model's training data incl. its
test split) or "new" (later rounds), so in-sample and out-of-sample errors are visible apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from grace_al import _tp, data, parity
from grace_al.models import ModelRef, fit_dir, fitted
from grace_al.state import round_dir, rounds_on_disk

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config

log = logging.getLogger("grace_al.properties")

PROPERTIES = ("energy", "relative", "adsorption", "cohesive")


class PropertyError(ValueError):
    pass


@dataclass
class Dataset:
    frames: list
    energy: np.ndarray
    forces: list
    round: np.ndarray                       # -1: not from a round (a file)
    strategy: list = field(default_factory=list)

    def subset(self, mask) -> Dataset:
        idx = np.flatnonzero(mask)
        return Dataset([self.frames[i] for i in idx], self.energy[idx],
                       [self.forces[i] for i in idx], self.round[idx],
                       [self.strategy[i] for i in idx] if self.strategy else [])


# ---------------------------------------------------------------- inputs
def labeled_data(cfg: Config) -> Dataset:
    """labeled.pkl.gz of every round (round index kept)."""
    frames, e, f, rnd, strat = [], [], [], [], []
    for k in rounds_on_disk(cfg.workdir):
        p = round_dir(cfg.workdir, k) / "labeled.pkl.gz"
        if not p.exists():
            continue
        df = pd.read_pickle(p)
        frames += list(df["ase_atoms"])
        e += list(df["energy"])
        f += list(df["forces"])
        rnd += [k] * len(df)
        strat += list(df["strategy"]) if "strategy" in df else [""] * len(df)
    if not frames:
        raise PropertyError(f"no labeled data in {cfg.workdir}")
    return Dataset(frames, np.array(e, dtype=float), f, np.array(rnd), strat)


def file_data(path: Path) -> Dataset:
    """Frames with DFT energies/forces (extxyz with a calculator, or a gracemaker pkl)."""
    if str(path).endswith((".pkl", ".pkl.gz")):
        df = pd.read_pickle(path)
        return Dataset(list(df["ase_atoms"]), df["energy"].to_numpy(dtype=float),
                       list(df["forces"]), np.full(len(df), -1))
    frames = data.read_frames(path)
    return Dataset(frames, np.array([at.get_potential_energy() for at in frames]),
                   [at.get_forces() for at in frames], np.full(len(frames), -1))


def resolve_model(cfg: Config, spec: str) -> tuple[ModelRef, int, str]:
    """(model with an artifact, last round it was trained on (-1: none), tag).
    spec: fm | last | round:K | K."""
    if spec == "fm":
        art = round_dir(cfg.workdir, 0) / "UQ" / "gmm_artifacts.npz"
        if not art.exists():
            raise PropertyError(f"{art} missing: run round 0's artifact stage first")
        m = _tp.fm_ref(cfg.foundation_model)
        return ModelRef("fm", m.model_yaml, m.checkpoint, art), -1, "fm"
    if spec == "last":
        done = [k for k in rounds_on_disk(cfg.workdir) if (fit_dir(cfg, k) / "FINISHED").exists()]
        if not done:
            raise PropertyError("no finished fit yet")
        spec = str(done[-1])
    k = int(spec.split(":")[-1])
    if not (fit_dir(cfg, k) / "FINISHED").exists():
        raise PropertyError(f"round {k} has no finished fit")
    m = fitted(cfg, k)
    art = m.artifact or round_dir(cfg.workdir, k + 1) / "UQ" / "gmm_artifacts.npz"
    if not art.exists():
        raise PropertyError(f"no UQ artifact for round {k}'s model (fit/UQ or round {k + 1}/UQ)")
    return ModelRef("finetuned", m.model_yaml, m.checkpoint, art), k, f"round{k:02d}"


def predict(model: ModelRef, frames: list, work: Path) -> tuple[np.ndarray, list]:
    """Energies and forces of `frames` (grace_uq predict in a subprocess)."""
    work.mkdir(parents=True, exist_ok=True)
    src = work / "frames.pkl.gz"
    pd.DataFrame({"ase_atoms": [at.copy() for at in frames]}).to_pickle(src)
    df = _tp.predict(model, src, work / "predicted.pkl.gz", log_file=work / "predict.log")
    return df["energy_predicted"].to_numpy(dtype=float), list(df["forces_predicted"])


def fit_mu(frames: list, energies) -> dict[str, float]:
    """mu_el from reference structures by least squares on their composition."""
    els, m = parity.compositions([at.get_chemical_symbols() for at in frames])
    mu = np.linalg.lstsq(m, np.asarray(energies, dtype=float), rcond=None)[0]
    return dict(zip(els, map(float, mu), strict=True))


# ---------------------------------------------------------------- properties
def _cell_key(at: Atoms) -> tuple:
    return tuple(np.round(np.asarray(at.cell).ravel(), 3))


def relative(frames, e_dft, e_pred) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E - E(DFT-lowest frame) within groups of identical composition and cell (>= 2)."""
    groups: dict[tuple, list[int]] = {}
    for i, at in enumerate(frames):
        groups.setdefault((at.get_chemical_formula(), _cell_key(at)), []).append(i)
    idx, x, y = [], [], []
    for members in groups.values():
        if len(members) < 2:
            continue
        ref = min(members, key=lambda i: e_dft[i])
        for i in members:
            if i != ref:
                idx.append(i)
                x.append(e_dft[i] - e_dft[ref])
                y.append(e_pred[i] - e_pred[ref])
    return np.array(idx, dtype=int), np.array(x), np.array(y)


def adsorption(frames, e_dft, e_pred, substrate: set[str], mu_dft=None, mu_pred=None):
    """Adsorption energies vs the clean slab of the same cell. Returns (indices, dft, model,
    adsorbate formulas, how mu was obtained)."""
    from collections import Counter

    clean: dict[tuple, int] = {}
    for i, at in enumerate(frames):
        if set(at.get_chemical_symbols()) <= substrate:
            clean.setdefault((len(at), _cell_key(at)), i)
    rows = []
    for i, at in enumerate(frames):
        sym = at.get_chemical_symbols()
        ads = Counter(s for s in sym if s not in substrate)
        if not ads:
            continue
        j = clean.get((len(sym) - sum(ads.values()), _cell_key(at)))
        if j is not None:
            rows.append((i, j, ads))
    if not rows:
        return np.zeros(0, int), np.zeros(0), np.zeros(0), [], "none"
    idx = np.array([i for i, _, _ in rows])
    raw_dft = np.array([e_dft[i] - e_dft[j] for i, j, _ in rows])
    raw_pred = np.array([e_pred[i] - e_pred[j] for i, j, _ in rows])
    els = sorted({e for _, _, a in rows for e in a})
    m = np.array([[a.get(e, 0) for e in els] for _, _, a in rows], dtype=float)
    how = "references"
    if mu_dft is None or mu_pred is None or not set(els) <= set(mu_dft) & set(mu_pred):
        how = "fitted"
        mu_dft = dict(zip(els, np.linalg.lstsq(m, raw_dft, rcond=None)[0], strict=True))
        mu_pred = dict(zip(els, np.linalg.lstsq(m, raw_pred, rcond=None)[0], strict=True))
    x = raw_dft - m @ np.array([mu_dft[e] for e in els])
    y = raw_pred - m @ np.array([mu_pred[e] for e in els])
    names = [adsorbate_label(a) for _, _, a in rows]
    return idx, x, y, names, how


def adsorbate_label(counts) -> str:
    """{"C": 2, "O": 2} -> "CO×2"; {"O": 1} -> "O"."""
    from functools import reduce
    from math import gcd

    g = reduce(gcd, counts.values())
    formula = "".join(f"{e}{n // g if n // g > 1 else ''}" for e, n in sorted(counts.items()))
    return formula + (f"×{g}" if g > 1 else "")


def cohesive(frames, e_dft, e_pred, mu_dft, mu_pred, min_vacuum: float = 6.0):
    """sum x_el mu_el - E/N (eV/atom) for frames without a vacuum gap (bulk)."""
    from grace_al.systems.base import vacuum_gap

    idx = [i for i, at in enumerate(frames) if at.pbc.all() and vacuum_gap(at) < min_vacuum]
    def coh(e, mu, i):
        sym = frames[i].get_chemical_symbols()
        return sum(mu[s] for s in sym) / len(sym) - e[i] / len(sym)
    missing = {s for i in idx for s in frames[i].get_chemical_symbols()} - set(mu_dft)
    if missing:
        raise PropertyError(f"cohesive energies need isolated-atom references for {missing} "
                            "(--references)")
    return (np.array(idx, dtype=int), np.array([coh(e_dft, mu_dft, i) for i in idx]),
            np.array([coh(e_pred, mu_pred, i) for i in idx]))


# ---------------------------------------------------------------- driver
def substrate_elements(cfg: Config, override: list[str] | None) -> set[str]:
    if override:
        return set(override)
    if cfg.system is not None and cfg.system.surface is not None:
        return {cfg.system.surface.element}
    raise PropertyError("adsorption: name the substrate elements (--substrate Pd)")


def run(cfg: Config, model_spec: str = "last", properties=("energy", "relative"),
        data_file: Path | None = None, references: Path | None = None,
        substrate: list[str] | None = None, out: Path | None = None) -> dict:
    """Parity plots (+ .npz) and metrics of the requested properties; returns the metrics."""
    model, trained, tag = resolve_model(cfg, model_spec)
    ds = file_data(data_file) if data_file else labeled_data(cfg)
    out = out or cfg.workdir / "report" / "validate"
    out.mkdir(parents=True, exist_ok=True)
    e_pred, f_pred = predict(model, ds.frames, out / f"pred_{tag}")
    seen = np.array(["trained on" if 0 <= r <= trained else "new" for r in ds.round])
    if data_file:
        seen = np.array([data_file.name] * len(ds.frames))
    mu_dft = mu_pred = None
    if references:
        refs = file_data(references)
        mu_dft = fit_mu(refs.frames, refs.energy)
        mu_pred = fit_mu(refs.frames, predict(model, refs.frames, out / f"pred_{tag}_refs")[0])
    who = "foundation model" if trained < 0 else f"model fine-tuned in round {trained}"
    results: dict = {"model": tag, "n_frames": len(ds.frames)}
    symbols = [at.get_chemical_symbols() for at in ds.frames]
    for prop in properties:
        png = out / f"{prop}_{tag}.png"
        if prop == "energy":
            d = parity.energy_force_data(symbols, ds.energy, e_pred, ds.forces, f_pred, seen)
            parity.plot_energy_forces(d, png, f"{who} vs DFT ({len(ds.frames)} structures)")
            results[prop] = {"energy": parity.metrics(d["e_dft"], d["e_pred"]),
                             "forces": parity.metrics(d["f_dft"], d["f_pred"])}
        else:
            if prop == "relative":
                idx, x, y = relative(ds.frames, ds.energy, e_pred)
                labels, unit, xl = seen[idx], "eV", "E - E(lowest)"
                note = " energies (same composition and cell)"
            elif prop == "adsorption":
                idx, x, y, names, how = adsorption(ds.frames, ds.energy, e_pred,
                                                   substrate_elements(cfg, substrate),
                                                   mu_dft, mu_pred)
                labels, unit, xl = np.array(names), "eV", "E_ads"
                note = " (mu fitted: up to a constant per adsorbate)" if how == "fitted" else ""
            elif prop == "cohesive":
                if mu_dft is None or mu_pred is None:
                    raise PropertyError("cohesive energies need --references (isolated atoms)")
                idx, x, y = cohesive(ds.frames, ds.energy, e_pred, mu_dft, mu_pred)
                labels, unit, xl, note = seen[idx], "eV/atom", "E_coh", ""
            else:
                raise PropertyError(f"unknown property {prop!r}; one of {PROPERTIES}")
            if not len(idx):
                log.warning("%s: no frames to compare", prop)
                results[prop] = {"n": 0}
                continue
            parity.plot_property(x, y, labels, png, f"{prop.capitalize()}{note}", xl, unit)
            d = {"dft": x, "model": y, "label": np.asarray(labels), "frame_index": idx,
                 "round": ds.round[idx]}
            results[prop] = parity.metrics(x, y)
        parity.save(png, d)
        results[prop]["plot"] = str(png)
    return results
