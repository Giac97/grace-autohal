"""Stage protocol and runner: done markers, config fingerprints, the stage registry.

A stage declares the config sections it depends on (`depends_on`). The runner hashes them
into a fingerprint and stores it in runs/round_XX/.done/<stage>.json after a successful
run. Next time: same fingerprint -> skip; different -> FingerprintMismatch (the outputs were
made with other settings; `grace-al redo` is the way around it). Versions are recorded in
the marker but not hashed, so upgrading grace-al or tensorpotential doesn't invalidate a
campaign.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from grace_al import __version__, _tp
from grace_al.state import now, read_marker, write_marker
from grace_al.utils import sha256_json

if TYPE_CHECKING:
    from grace_al.config.schema import Config
    from grace_al.models import ModelRef
    from grace_al.systems.base import SystemInfo

log = logging.getLogger("grace_al.stages")

# stage name -> implementation (tasks 3 and 4 replace the interim seeds/explore/label)
STAGE_CLASSES = {
    "seeds": "grace_al.stages.seeds:SeedsStage",
    "artifact": "grace_al.stages.artifact:ArtifactStage",
    "explore": "grace_al.stages.explore:ExploreStage",
    "select": "grace_al.stages.select:SelectStage",
    "label": "grace_al.stages.label:LabelStage",
    "validate": "grace_al.stages.validate:ValidateStage",
    "finetune": "grace_al.stages.finetune:FinetuneStage",
}


@dataclass
class RoundContext:
    cfg: Config
    k: int
    root: Path                        # runs/
    rdir: Path                        # runs/round_XX (created)
    system: SystemInfo
    driver: ModelRef


@dataclass
class StageResult:
    # failed: the stage ran but its result is unusable (message in summary["error"])
    status: Literal["done", "waiting", "converged", "skipped", "failed"]
    summary: dict = field(default_factory=dict)   # stored in the done marker, shown by `report`


class Stage(Protocol):
    name: str
    depends_on: tuple[str, ...]       # config sections for the fingerprint

    def applies(self, ctx: RoundContext) -> bool: ...

    def run(self, ctx: RoundContext) -> StageResult: ...


class FingerprintMismatch(RuntimeError):
    def __init__(self, stage: str, k: int, old: str | None, new: str | None):
        self.stage, self.round, self.old, self.new = stage, k, old, new
        super().__init__(
            f"round {k}, stage {stage}: the config sections it depends on changed since it "
            f"ran (fingerprint {str(old)[:12]} -> {str(new)[:12]}). Restore the old settings, "
            f"or recompute from there: grace-al redo --round {k} --from {stage}")


class StageFailed(RuntimeError):
    def __init__(self, stage: str, k: int, message: str):
        self.stage, self.round = stage, k
        super().__init__(f"round {k}, stage {stage} failed: {message}")


def get_stage(name: str) -> Stage:
    mod, cls = STAGE_CLASSES[name].split(":")
    return getattr(importlib.import_module(mod), cls)()


def fingerprint(stage: Stage, ctx: RoundContext) -> str | None:
    """Hash of the stage's config sections (+ its `fingerprint_extra(ctx)`, if it has one);
    None for a stage without dependencies (never mismatches)."""
    extra = getattr(stage, "fingerprint_extra", None)
    extra = extra(ctx) if extra else None
    if not stage.depends_on and extra is None:
        return None
    return sha256_json({"sections": ctx.cfg.section_fingerprint(*stage.depends_on),
                        "extra": extra})


def versions() -> dict:
    return {"grace_al": __version__, "tensorpotential": _tp.tp_version()}


def run_stage(stage: Stage, ctx: RoundContext) -> StageResult:
    """Skip (not applicable, or done with this fingerprint), refuse (done with another
    fingerprint), or run and mark done. A skipped stage that had ended the loop
    (converged) reports that again, so a re-run stops at the same place."""
    if not stage.applies(ctx):
        return StageResult("skipped")
    fp = fingerprint(stage, ctx)
    marker = read_marker(ctx.rdir, stage.name)
    if marker is not None:
        if fp is not None and marker.get("fingerprint") != fp:
            raise FingerprintMismatch(stage.name, ctx.k, marker.get("fingerprint"), fp)
        log.debug("round %d %s: done before, skipped", ctx.k, stage.name)
        return StageResult("converged" if marker.get("result") == "converged" else "skipped",
                           marker.get("summary", {}))
    started = now()
    log.info("round %d: %s", ctx.k, stage.name)
    res = stage.run(ctx)
    if res.status == "failed":
        raise StageFailed(stage.name, ctx.k, str(res.summary.get("error", "see the log")))
    if res.status in ("done", "converged"):
        summary = dict(res.summary)
        write_marker(ctx.rdir, stage.name, {
            "stage": stage.name, "round": ctx.k, "fingerprint": fp, "result": res.status,
            "versions": {**versions(), **summary.pop("versions", {})},
            "started": started, "finished": now(), "summary": summary})
    return res
