"""Atoms <-> gracemaker DataFrames, the training set of a round, element counts."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ase import Atoms

    from grace_al.config.schema import Config

log = logging.getLogger("grace_al.data")

COLUMNS = ("ase_atoms", "energy", "forces", "stress", "energy_corrected")


def to_dataframe(frames: list[Atoms]) -> pd.DataFrame:
    """ase.Atoms with energy/forces[/stress] (SinglePointCalculator) -> gracemaker DataFrame.
    `stress` (3x3) only if every frame has it; calculators are removed."""
    rows, stresses = [], []
    for at in frames:
        rows.append({"ase_atoms": at.copy(),
                     "energy": at.get_potential_energy(),
                     "forces": at.get_forces()})
        try:
            stresses.append(at.get_stress(voigt=False))
        except Exception:
            stresses.append(None)
    df = pd.DataFrame(rows, columns=["ase_atoms", "energy", "forces"])
    if frames and all(s is not None for s in stresses):
        df["stress"] = stresses
    df["energy_corrected"] = df["energy"]
    for at in df["ase_atoms"]:
        at.calc = None
    return df


def read_frames(path) -> list[Atoms]:
    """All frames of a structure file as a list (empty file -> [])."""
    from pathlib import Path

    from ase.io import read

    if not Path(path).stat().st_size:
        return []
    frames = read(path, ":")
    return frames if isinstance(frames, list) else [frames]


def placeholder_frame(seeds: list[Atoms]) -> pd.DataFrame:
    """Seeds with zero labels: the round-0 artifact only uses features."""
    return pd.DataFrame({"ase_atoms": [at.copy() for at in seeds],
                         "energy": 0.0, "energy_corrected": 0.0,
                         "forces": [np.zeros((len(at), 3)) for at in seeds]})


def structure_hash(at: Atoms, decimals: int = 6) -> str:
    """Same numbers, cell and positions (rounded to 10^-decimals Å) -> same hash."""
    h = hashlib.sha1()
    r = lambda a: np.round(np.asarray(a, dtype=np.float64), decimals) + 0.0  # no -0.0
    h.update(np.asarray(at.numbers, dtype=np.int64).tobytes())
    h.update(r(at.cell).tobytes())
    h.update(r(at.positions).tobytes())
    return h.hexdigest()


def dataset_hash(df: pd.DataFrame) -> str:
    """Order-sensitive hash of the structures of a DataFrame (not of the labels)."""
    h = hashlib.sha256()
    for at in df["ase_atoms"]:
        h.update(structure_hash(at).encode())
    return h.hexdigest()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    keys = df["ase_atoms"].map(structure_hash)
    out = df[~keys.duplicated()].reset_index(drop=True)
    if len(out) < len(df):
        log.info("training set: %d duplicate structures dropped", len(df) - len(out))
    return out


def training_set(cfg: Config, k: int) -> pd.DataFrame | None:
    """All labeled data available at the start of round k (labeled.pkl.gz of rounds < k),
    duplicates dropped; None if there is none."""
    dfs = []
    for j in range(k):
        p = cfg.workdir / f"round_{j:02d}" / "labeled.pkl.gz"
        if p.exists():
            dfs.append(pd.read_pickle(p))
    dfs = [d for d in dfs if len(d)]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    if "stress" in df.columns and df["stress"].isna().any():
        df = df.drop(columns="stress")      # some rounds without stress: drop it for all
    return drop_duplicates(df)


def element_counts(df: pd.DataFrame) -> dict[str, int]:
    """Atoms per element over all structures."""
    c: Counter[str] = Counter()
    for at in df["ase_atoms"]:
        c.update(at.get_chemical_symbols())
    return dict(sorted(c.items()))
