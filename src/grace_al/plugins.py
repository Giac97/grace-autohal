"""Plug-in registry: entry point groups grace_al.systems and grace_al.labelers."""

from __future__ import annotations

from importlib import metadata


class PluginError(LookupError):
    pass


def names(group: str) -> list[str]:
    return sorted(ep.name for ep in metadata.entry_points(group=group))


def load(group: str, name: str):
    """The object an entry point names (a class, normally)."""
    eps = {ep.name: ep for ep in metadata.entry_points(group=group)}
    if name not in eps:
        raise PluginError(f"no {group} plug-in {name!r}; installed: {', '.join(sorted(eps))}")
    try:
        return eps[name].load()
    except (ImportError, AttributeError) as err:
        raise PluginError(f"{group} plug-in {name!r} ({eps[name].value}) cannot be loaded: "
                          f"{err}") from err
