import builtins

import pytest

from tradingagents_investboard import _framework


def test_the_installed_framework_is_recognised():
    # The dev environment installs TauricResearch's framework from its tag.
    assert _framework.missing() is None


def test_no_framework_at_all_names_the_install_command(monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("tradingagents.") or name == "tradingagents":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    message = _framework.missing()

    assert message is not None
    assert _framework.INSTALL_COMMAND in message
    assert "TauricResearch" in message


def test_the_pypi_namesake_is_told_apart_from_the_framework(monkeypatch):
    # A package called `tradingagents` that lacks the vendor registry is the
    # unrelated project that owns the name on PyPI, not the framework.
    import types

    impostor = types.ModuleType("tradingagents.dataflows.interface")
    monkeypatch.setattr(_framework, "_load_interface", lambda: impostor)

    message = _framework.missing()

    assert message is not None
    assert "different project" in message
    assert _framework.INSTALL_COMMAND in message


def test_require_raises_with_the_same_message(monkeypatch):
    monkeypatch.setattr(_framework, "missing", lambda: "not here")
    with pytest.raises(_framework.FrameworkMissing, match="not here"):
        _framework.require()
