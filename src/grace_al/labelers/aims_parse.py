"""Reading FHI-aims output: results, version, SCF iterations, and why a run failed.

Kept apart from aims.py (inputs, running) so that a VASP labeler can follow the same split.
"""

from __future__ import annotations

import re
from pathlib import Path

FINISHED = "Have a nice day"

# category -> what usually fixes it (shown by the label stage and in failures.json)
HINTS = {
    "stacksize": "the stack limit is too low for aims: the launcher sets it to unlimited when "
                 "the hard limit allows it; check `ulimit -Hs` in the job (ask the admins if "
                 "it is limited)",
    "mpi": "aims was started with the wrong MPI or none: the grace env's Intel MPI must not "
           "leak into aims. Save the module env in the job (runs/.aims_env, written before "
           "the Python env is activated) or set hpc.mpirun / hpc.env (site profile)",
    "charge_integration": "wrong density from MPI collectives or the ELSI DM builder "
                          "(FHI-aims manual, 'Charge integration errors'): set "
                          "aims.parameters.elsi_build_dm_method: pdgemm and the Intel MPI "
                          "settings I_MPI_COLL_INTRANODE=pt2pt, I_MPI_PIN_DOMAIN=1, "
                          "I_MPI_PIN_CELL=core (site profile bwunicluster3)",
    "scf_not_converged": "raise aims.parameters.sc_iter_limit, or use a gentler mixing "
                         "(charge_mix_param 0.05) or more smearing; metal slabs need 60-80 "
                         "iterations with light settings",
    "setup": "aims stopped before the SCF: read the *** lines in aims.out (species files, "
             "control.in keywords, geometry)",
    "crashed": "aims stopped without an error message (killed, time limit, out of memory?): "
               "see aims.err and the job log",
    "parse": "aims finished, but its output could not be read",
}

_OUT_ERRORS = (   # (category, pattern in aims.out), first match wins
    ("stacksize", re.compile(r"stacksize too low", re.I)),
    ("charge_integration", re.compile(r"Charge integration error is too large", re.I)),
    ("scf_not_converged", re.compile(r"SCF cycle not converged", re.I)),
)
_MPI_ERR = re.compile(
    r"prterun|prte|libfabric|fi_getinfo|PMI[ _2x]|MPI_Init|mpirun was unable|"
    r"mpirun: command not found|mpiexec|hydra|orted|not enough slots|"
    r"error while loading shared libraries", re.I)
_SCF_STEP = "Begin self-consistency iteration #"


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def finished(text: str) -> bool:
    return FINISHED in text[-4000:]


def version(text: str) -> str | None:
    """'261007' (or '261007+<commit>' when the build knows its commit)."""
    m = re.search(r"FHI-aims version\s*:\s*(\S+)", text)
    if not m:
        return None
    c = re.search(r"Commit number\s*:\s*(\S+)", text)
    return m.group(1) + (f"+{c.group(1)}" if c and "NOTFOUND" not in c.group(1) else "")


def scf_iterations(text: str) -> int:
    """Number of the last SCF iteration started (0 if none)."""
    found = re.findall(re.escape(_SCF_STEP) + r"\s*(\d+)", text)
    return int(found[-1]) if found else 0


def _first_error(text: str) -> str:
    """The first '***' line (aims's error marker), stripped."""
    for line in text.splitlines():
        if "***" in line and line.strip("* \t"):
            return line.strip().lstrip("* ").strip()
    return ""


def categorize(out: str, err: str) -> tuple[str, str]:
    """(category, message) of an unfinished aims run from the text of aims.out and aims.err."""
    if "grace-al: interrupted" in err:
        return "crashed", "interrupted (SIGTERM: time limit or scancel); runs again on restart"
    for cat, pat in _OUT_ERRORS:
        m = pat.search(out)
        if m:
            line = out[out.rfind("\n", 0, m.start()) + 1: out.find("\n", m.end())]
            return cat, line.strip().lstrip("* ").strip()
    # MPI: aims never really started (no header), or the launcher complained
    if "Invoking FHI-aims" not in out or not out.strip():
        m = _MPI_ERR.search(err)
        if m:
            line = err[err.rfind("\n", 0, m.start()) + 1: err.find("\n", m.end())]
            return "mpi", line.strip()
    if _SCF_STEP not in out:
        msg = _first_error(out) or _last_line(err) or "no SCF iteration in aims.out"
        if "Invoking FHI-aims" not in out:
            return "crashed", msg if msg else "aims.out is empty"
        return "setup", msg
    return "crashed", (_first_error(out) or _last_line(err)
                       or f"stopped after {scf_iterations(out)} SCF iterations "
                          "without an error message")


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def read_results(out_path: Path, pbc=None):
    """Atoms + SinglePointCalculator (energy, forces[, stress]) from a finished aims.out;
    `pbc` (from the input frame) replaces what the parser guessed."""
    from ase import Atoms
    from ase.io import read

    at = read(out_path, format="aims-output")
    assert isinstance(at, Atoms)
    if pbc is not None:
        at.pbc = pbc
    return at
