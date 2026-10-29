"""Which SLURM partitions the current user may submit to (`grace-al prepare`).

"sbatch: error: Batch job submission failed: User's group not permitted to use this
partition" comes from AllowGroups / AllowAccounts; this reads them with scontrol and
compares them with `id -Gn` and the user's accounts (sacctmgr, when available).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class Partition:
    name: str
    allow_groups: list[str] = field(default_factory=list)     # ["ALL"] = everyone
    allow_accounts: list[str] = field(default_factory=list)
    max_time: str = ""
    tres: str = ""
    open: bool = False


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


def parse_scontrol(text: str) -> list[Partition]:
    """`scontrol show partition` (records separated by blank lines)."""
    parts = []
    for block in re.split(r"\n\s*\n", text.strip()):
        kv = dict(re.findall(r"(\w+)=(\S*)", block))
        if "PartitionName" not in kv:
            continue
        split = lambda v: [x for x in v.split(",") if x] if v else ["ALL"]
        parts.append(Partition(kv["PartitionName"], split(kv.get("AllowGroups", "ALL")),
                               split(kv.get("AllowAccounts", "ALL")), kv.get("MaxTime", ""),
                               kv.get("TRES", "")))
    return parts


def user_groups() -> set[str]:
    return set(_run(["id", "-Gn"]).split())


def user_accounts() -> set[str] | None:
    """The user's SLURM accounts, or None if sacctmgr is not available."""
    if not shutil.which("sacctmgr"):
        return None
    import getpass

    out = _run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={getpass.getuser()}",
                "format=Account"])
    return {a.strip() for a in out.splitlines() if a.strip()}


def mark_open(parts: list[Partition], groups: set[str],
              accounts: set[str] | None) -> list[Partition]:
    for p in parts:
        g_ok = "ALL" in p.allow_groups or bool(groups & set(p.allow_groups))
        a_ok = ("ALL" in p.allow_accounts or accounts is None
                or bool(accounts & set(p.allow_accounts)))
        p.open = g_ok and a_ok
    return parts


def partitions() -> list[Partition] | None:
    """All partitions with `open` set for this user; None without scontrol."""
    if not shutil.which("scontrol"):
        return None
    return mark_open(parse_scontrol(_run(["scontrol", "show", "partition"])), user_groups(),
                     user_accounts())
