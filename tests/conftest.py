import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """No GRACE_AL__ overrides from the shell; an empty model cache."""
    import os

    for key in list(os.environ):
        if key.startswith("GRACE_AL__"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GRACE_CACHE", str(tmp_path / "grace_cache"))
