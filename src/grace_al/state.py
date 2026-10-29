"""Campaign state on disk: runs/state.json, per-stage done markers, the output table, redo.

runs/state.json                    {round, stage, status, updated, jobs}
runs/round_XX/.done/<stage>.json   {stage, fingerprint, result, versions, started, finished,
                                    summary}; written after a successful stage run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger("grace_al.state")

Status = Literal["running", "waiting_for_labels", "converged", "finished", "failed"]

# order within a round (seeds: round 0 only; validate: from round 1)
STAGES = ("seeds", "artifact", "explore", "select", "label", "validate", "finetune")

# what each stage writes, relative to runs/round_XX (redo deletes these; report reads them).
# seeds: the seeds file lives next to config.yaml and is never deleted by redo.
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "seeds": (),
    "artifact": ("artifact_data.pkl.gz", "UQ"),
    "explore": ("hal", "starts.extxyz"),
    "select": ("selected.pkl.gz", "selected.extxyz", "select.log"),
    "label": ("to_label.extxyz", "dft", "labeled.pkl.gz", "failures.json",
              "MANUAL_LABELING.txt"),
    "validate": ("validation.json", "validate"),
    "finetune": ("fit",),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def round_dir(root: Path, k: int) -> Path:
    return root / f"round_{k:02d}"


def rounds_on_disk(root: Path) -> list[int]:
    return sorted(int(p.name[6:]) for p in root.glob("round_[0-9]*") if p.name[6:].isdigit())


def write_json_atomic(path: Path, obj: Any) -> None:
    """Temporary file in the same folder + os.replace: readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- state.json
@dataclass
class State:
    round: int = 0
    stage: str | None = None
    status: Status = "running"
    updated: str = ""
    jobs: list[dict] = field(default_factory=list)   # SLURM jobs (task 5)
    message: str = ""                                # reason for failed / finished


def state_path(root: Path) -> Path:
    return root / "state.json"


def load_state(root: Path) -> State:
    p = state_path(root)
    if not p.exists():
        return State()
    d = json.loads(p.read_text())
    return State(**{k: v for k, v in d.items() if k in State.__dataclass_fields__})


def save_state(root: Path, state: State) -> State:
    state.updated = now()
    write_json_atomic(state_path(root), asdict(state))
    return state


def update_state(root: Path, **changes: Any) -> State:
    st = load_state(root)
    for k, v in changes.items():
        setattr(st, k, v)
    return save_state(root, st)


# ---------------------------------------------------------------- done markers
def marker_path(rdir: Path, stage: str) -> Path:
    return rdir / ".done" / f"{stage}.json"


def read_marker(rdir: Path, stage: str) -> dict | None:
    p = marker_path(rdir, stage)
    return json.loads(p.read_text()) if p.exists() else None


def write_marker(rdir: Path, stage: str, marker: dict) -> None:
    write_json_atomic(marker_path(rdir, stage), marker)


# ---------------------------------------------------------------- redo
def redo_targets(root: Path, k: int, from_stage: str) -> list[Path]:
    """Existing paths `redo` deletes: markers + outputs of `from_stage` and the later stages
    of round k, every later round, and runs/FINISHED."""
    if from_stage not in STAGES:
        raise ValueError(f"unknown stage {from_stage!r}; stages: {', '.join(STAGES)}")
    rdir = round_dir(root, k)
    paths = []
    for s in STAGES[STAGES.index(from_stage):]:
        paths.append(marker_path(rdir, s))
        paths += [rdir / p for p in STAGE_OUTPUTS[s]]
    paths += [round_dir(root, j) for j in rounds_on_disk(root) if j > k]
    paths.append(root / "FINISHED")
    return [p for p in paths if p.exists() or p.is_symlink()]


def redo(root: Path, k: int, from_stage: str, yes: bool = False,
         confirm: Callable[[list[Path]], bool] | None = None) -> list[Path]:
    """Delete what `redo_targets` lists so the next `run` recomputes it.
    Without `yes`, `confirm(paths)` decides (default: ask on the terminal).
    Returns the deleted paths ([] if declined or nothing to do)."""
    paths = redo_targets(root, k, from_stage)
    if not paths:
        return []
    if not yes:
        confirm = confirm or _ask
        if not confirm(paths):
            return []
    for p in paths:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        log.info("deleted %s", p)
    if state_path(root).exists():
        update_state(root, round=k, stage=from_stage, status="running",
                     message=f"redo round {k} from {from_stage}")
    return paths


def _ask(paths: list[Path]) -> bool:
    print("will delete:")
    for p in paths:
        print(f"  {p}")
    return input("proceed? [y/N] ").strip().lower() in ("y", "yes")
