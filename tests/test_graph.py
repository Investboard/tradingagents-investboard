import json
import os
import stat

from tradingagents_investboard import graph


def test_an_outbox_entry_is_owner_only(tmp_path, monkeypatch):
    """A failed post parks the whole run payload on disk, so both the file and its directory are owner-only."""
    monkeypatch.setattr(graph, "OUTBOX_DIR", tmp_path / "outbox")

    path = graph.write_outbox({"framework_run_id": "run-1", "ticker": "SAP.DE"})

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert json.loads(path.read_text(encoding="utf-8"))["framework_run_id"] == "run-1"
