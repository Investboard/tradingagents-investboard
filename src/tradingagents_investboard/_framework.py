"""Whether the TradingAgents framework is installed, and what to run when it is not.

The framework is NOT a declared dependency of this package, on purpose. It is
not published on PyPI: the PyPI project named ``tradingagents`` is an unrelated
one, so declaring that name would install a stranger's code, and PyPI refuses
the direct Git reference that would name the right one. Whoever runs
TradingAgents already has it, from the clone they run it from, so this package
checks for it when ``analyze`` or ``replay`` starts and says exactly what to do
otherwise. ``connect`` and ``status`` need no framework and never ask.
"""

from __future__ import annotations

from types import ModuleType

# The tag this release is tested against.
FRAMEWORK_TAG = "v0.4.0"
INSTALL_COMMAND = (
    "pip install git+https://github.com/TauricResearch/TradingAgents.git@" + FRAMEWORK_TAG
)


class FrameworkMissing(RuntimeError):
    """The framework is absent, or the installed ``tradingagents`` is not it."""


def _load_interface() -> ModuleType:
    import tradingagents.dataflows.interface as interface

    return interface


def missing() -> str | None:
    """One actionable sentence when the framework cannot be used, else None."""
    try:
        interface = _load_interface()
    except ImportError:
        return (
            "TradingAgents is not installed. It is TauricResearch's framework and is "
            f"not on PyPI; install it with: {INSTALL_COMMAND}"
        )
    if not hasattr(interface, "VENDOR_METHODS") or not hasattr(interface, "VENDOR_LIST"):
        return (
            "The installed package named 'tradingagents' is a different project from "
            "TauricResearch's framework (the PyPI name belongs to someone else). "
            f"Remove it and install the framework with: {INSTALL_COMMAND}"
        )
    return None


def require() -> None:
    """Raise ``FrameworkMissing`` with the sentence ``missing`` would give."""
    message = missing()
    if message is not None:
        raise FrameworkMissing(message)
