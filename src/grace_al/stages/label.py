"""label: the selected frames (round 0: also the seeds) through the configured labeler
-> labeled.pkl.gz (+ failures.json).

Modes (label.mode):
  inline      one calculation after another, each with all ranks ($AIMS_NP / $SLURM_NTASKS
              / aims.np)
  concurrent  label.concurrent_calcs calculations side by side, each with
              ranks // concurrent_calcs ranks on its own consecutive cores
              (I_MPI_PIN_PROCESSOR_LIST for Intel MPI; OpenMPI binding is switched off so
              that the runs don't all pin to cores 0..n-1)
  array       prepare every task, list the ones not started in dft/tasks.txt and wait
              (exit code 3); a CPU job array runs them (`grace-al aims-run <dir>`, task 5),
              and the next `grace-al run` collects

labeled.pkl.gz has the columns of data.to_dataframe plus `strategy` (calm-greedy,
random-all, seed) and `label_index` (row of to_label.extxyz). The fingerprint covers the
aims section and the labeler name/model, not the execution mode: switching from inline to
array mid-campaign is fine, changing DFT settings is not.
"""

from __future__ import annotations

import logging
import queue
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grace_al import data
from grace_al.labelers import aims_parse
from grace_al.labelers.base import (
    Labeler,
    LabelFailure,
    LabelTask,
    Slot,
    get_labeler,
    write_failures,
)
from grace_al.stages.base import RoundContext, StageResult

log = logging.getLogger("grace_al.stages.label")


def frames_to_label(ctx: RoundContext) -> list:
    """selected.extxyz (+ the seeds in round 0), with info strategy and label_index."""
    sel = ctx.rdir / "selected.extxyz"
    frames = data.read_frames(sel) if sel.exists() else []
    if ctx.k == 0:
        seeds = data.read_frames(ctx.cfg.seeds)
        for at in seeds:
            at.info["strategy"] = "seed"
        frames += seeds
    out = []
    for i, at in enumerate(frames):
        at = at.copy()
        at.calc = None
        at.info = {"strategy": at.info.get("strategy", ""), "label_index": i}
        out.append(at)
    return out


def slots(cfg, n: int) -> list[Slot]:
    """concurrent mode: n slots, each with its share of the ranks on consecutive cores."""
    from grace_al.labelers.launcher import total_ranks

    r = max(1, total_ranks(cfg) // n)
    return [Slot(r, {"I_MPI_PIN_PROCESSOR_LIST": f"{s * r}-{s * r + r - 1}",
                     "OMPI_MCA_hwloc_base_binding_policy": "none"}) for s in range(n)]


def run_concurrent(labeler: Labeler, tasks: list[LabelTask], free: list[Slot]) -> None:
    pool_slots: queue.Queue[Slot] = queue.Queue()
    for s in free:
        pool_slots.put(s)

    def one(task):
        slot = pool_slots.get()
        try:
            labeler.run(task, slot)
        finally:
            pool_slots.put(slot)

    with ThreadPoolExecutor(len(free)) as ex:
        for f in [ex.submit(one, t) for t in tasks]:
            f.result()


def started(labeler: Labeler, task: LabelTask) -> bool:
    fn = getattr(labeler, "started", None)
    return fn(task) if fn else labeler.is_finished(task)


class LabelStage:
    name = "label"
    depends_on = ("aims",)

    def fingerprint_extra(self, ctx: RoundContext) -> dict:
        return {"labeler": ctx.cfg.label.labeler, "teacher_model": ctx.cfg.label.teacher_model}

    def applies(self, ctx: RoundContext) -> bool:
        return True

    def run(self, ctx: RoundContext) -> StageResult:
        from ase.io import write

        cfg, mode = ctx.cfg, ctx.cfg.label.mode
        frames = frames_to_label(ctx)
        write(ctx.rdir / "to_label.extxyz", frames, format="extxyz")
        labeler = get_labeler(cfg)
        dft = ctx.rdir / "dft"
        tasks = labeler.prepare(frames, dft, ctx.system)
        pending = [t for t in tasks if not labeler.is_finished(t)]
        log.info("round %d: %d structures to label (%d done before), labeler %s, mode %s",
                 ctx.k, len(tasks), len(tasks) - len(pending), labeler.name, mode)

        if pending and labeler.name == "manual":
            log.warning("round %d: waiting for labels; see %s", ctx.k,
                        ctx.rdir / "MANUAL_LABELING.txt")
            return StageResult("waiting", {"n_tasks": len(tasks), "n_pending": len(pending)})
        if pending and mode == "array":
            todo = [t for t in pending if not started(labeler, t)]
            if todo:
                listing = dft / "tasks.txt"
                listing.parent.mkdir(parents=True, exist_ok=True)
                listing.write_text("".join(f"{t.workdir}\n" for t in todo))
                log.info("round %d: %d tasks listed in %s for the job array", ctx.k, len(todo),
                         listing)
                return StageResult("waiting", {"n_tasks": len(tasks), "n_pending": len(todo),
                                               "tasks_file": str(listing)})
        elif pending and mode == "concurrent":
            run_concurrent(labeler, pending, slots(cfg, min(cfg.label.concurrent_calcs,
                                                           len(pending))))
        elif pending:
            batch = getattr(labeler, "run_batch", None)
            if batch:
                batch(pending)
            else:
                for t in pending:
                    labeler.run(t)
        return self._collect(ctx, labeler, tasks, frames)

    @staticmethod
    def _collect(ctx: RoundContext, labeler: Labeler, tasks: list[LabelTask],
                 frames: list) -> StageResult:
        labeled, failures = [], []
        for t in tasks:
            res = labeler.collect(t)
            if isinstance(res, LabelFailure):
                failures.append(res)
            else:
                res.info.update(frames[t.index].info)
                labeled.append(res)
        write_failures(ctx.rdir / "failures.json", failures)
        counts = Counter(f.category for f in failures)
        summary: dict = {"labeler": labeler.name, "mode": ctx.cfg.label.mode,
                         "n_tasks": len(tasks), "n_labeled": len(labeled),
                         "n_failed": len(failures), "failures": dict(counts.most_common())}
        extra = getattr(labeler, "summary", None)
        if extra:
            summary.update(extra(tasks))
        log.info("round %d: %d/%d structures labeled%s", ctx.k, len(labeled), len(tasks),
                 f", failures {summary['failures']}" if failures else "")
        if tasks and len(failures) / len(tasks) > ctx.cfg.label.max_failure_fraction:
            return StageResult("failed", {**summary, "error": failure_message(
                ctx.k, ctx.rdir, failures, len(tasks), ctx.cfg.label.max_failure_fraction)})
        df = data.to_dataframe(labeled)
        df["strategy"] = [at.info.get("strategy", "") for at in labeled]
        df["label_index"] = [at.info.get("label_index", -1) for at in labeled]
        tmp = ctx.rdir / ".labeled.pkl.gz"
        df.to_pickle(tmp)
        tmp.replace(ctx.rdir / "labeled.pkl.gz")
        return StageResult("done", summary)


def failure_message(k: int, rdir: Path, failures: list[LabelFailure], n: int,
                    limit: float) -> str:
    cat, count = Counter(f.category for f in failures).most_common(1)[0]
    example = next(f for f in failures if f.category == cat)
    return (f"{len(failures)}/{n} labels failed (> label.max_failure_fraction {limit}). "
            f"Most common: {cat} ({count}x), e.g. task {example.index:04d}: {example.message}. "
            f"Fix: {aims_parse.HINTS.get(cat, 'see ' + str(rdir / 'failures.json'))}. "
            f"Details in {rdir / 'failures.json'}; after fixing: "
            f"grace-al redo --round {k} --from label")
