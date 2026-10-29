from pathlib import Path

from grace_al.config import docs

DOCS = Path(__file__).parents[2] / "docs" / "config.md"


def test_every_key_described():
    missing = [docs.key_path(name, key) for name, model in docs.SECTIONS
               for key, f in model.model_fields.items()
               if not f.exclude and not docs.description(name, model, key)]
    assert missing == [], "describe these keys (schema.py comment or docs.DESCRIPTIONS)"


def test_no_stale_descriptions():
    keys = {docs.key_path(n, k) for n, m in docs.SECTIONS for k in m.model_fields}
    assert set(docs.DESCRIPTIONS) <= keys


def test_config_md_up_to_date():
    assert DOCS.read_text() == docs.render(), \
        "regenerate: python -m grace_al.config.docs > docs/config.md"
