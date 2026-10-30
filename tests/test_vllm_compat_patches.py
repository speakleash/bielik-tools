"""Backport semantics and filesystem isolation; no vLLM/GPU dependency."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools" / "patch_vllm_compat.py"
spec = importlib.util.spec_from_file_location("compat_backports", PATH)
patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patches)


def effort_source():
    return (
        "def prepare(request):\n    if True:\n        if True:\n            if True:\n"
        + patches.EFFORT_OLD
        + "\n                return chat_template_kwargs\n"
    )


@pytest.mark.parametrize(
    "kwargs,explicit,expected",
    [
        ({"reasoning_effort": "none"}, None, {"reasoning_effort": "none"}),
        (
            {"reasoning_effort": "low", "enable_thinking": True},
            None,
            {"reasoning_effort": "low", "enable_thinking": True},
        ),
        ({"reasoning_effort": "low"}, "high", {"reasoning_effort": "high"}),
        (None, None, {}),
        (None, "medium", {"reasoning_effort": "medium"}),
    ],
)
def test_effort_precedence_preserves_template_value_when_api_field_is_unset(
    kwargs, explicit, expected
):
    source, changed = patches.transform(
        effort_source(), patches.EFFORT_OLD, patches.EFFORT_NEW
    )
    namespace = {}
    exec(source, namespace)
    before = dict(kwargs) if kwargs is not None else None
    request = SimpleNamespace(chat_template_kwargs=kwargs, reasoning_effort=explicit)
    assert namespace["prepare"](request) == expected
    if explicit is None:
        assert request.chat_template_kwargs == before
    assert changed


@pytest.mark.parametrize(
    "old,new,source",
    [
        (patches.EFFORT_OLD, patches.EFFORT_NEW, effort_source()),
    ],
)
def test_backports_are_idempotent(old, new, source):
    patched, changed = patches.transform(source, old, new)
    assert changed
    repeated, changed = patches.transform(patched, old, new)
    assert not changed
    assert repeated == patched


def test_apply_breaks_file_symlink_without_modifying_shared_cache(tmp_path):
    cache = tmp_path / "cached.py"
    cache.write_text(effort_source())
    target = tmp_path / "environment.py"
    target.symlink_to(cache)
    assert "needed" in patches.apply_file(
        target, patches.EFFORT_OLD, patches.EFFORT_NEW, apply=False
    )
    assert target.is_symlink()
    patches.apply_file(target, patches.EFFORT_OLD, patches.EFFORT_NEW, apply=True)
    assert not target.is_symlink()
    assert cache.read_text() == effort_source()
    assert patches.EFFORT_NEW in target.read_text()
    assert (
        target.with_name(target.name + ".bielik-compat.orig").read_text()
        == effort_source()
    )
    assert (
        patches.apply_file(target, patches.EFFORT_OLD, patches.EFFORT_NEW, apply=True)
        == "already applied"
    )


def test_refuse_unrecognized_source_without_creating_backup(tmp_path):
    target = tmp_path / "unknown.py"
    target.write_text("x = 1\n")
    with pytest.raises(ValueError, match="Unexpected"):
        patches.apply_file(target, patches.EFFORT_OLD, patches.EFFORT_NEW, apply=True)
    assert target.read_text() == "x = 1\n"
    assert list(tmp_path.iterdir()) == [target]


def test_refuse_symlinked_parent_directory(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "module.py").write_text(effort_source())
    link = tmp_path / "package"
    link.symlink_to(shared, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked package"):
        patches.apply_file(
            link / "module.py", patches.EFFORT_OLD, patches.EFFORT_NEW, apply=True
        )
    assert (shared / "module.py").read_text() == effort_source()


def test_vllm_020_is_not_patched(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["patch_vllm_compat.py", "--apply"])
    monkeypatch.setattr(
        patches.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(version="0.20.0"),
    )

    def unexpected_write(*args, **kwargs):
        raise AssertionError("vLLM 0.20 must use the native plugin")

    monkeypatch.setattr(patches, "apply_file", unexpected_write)
    patches.main()
    assert "unchanged" in capsys.readouterr().out
