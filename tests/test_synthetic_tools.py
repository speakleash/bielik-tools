import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "synthetic_tools.py"
spec = importlib.util.spec_from_file_location("synthetic_tools_test", PATH)
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


@pytest.mark.parametrize("name,args", [
    ("shell", {"command": "touch file"}), ("add", {"a": True, "b": 2}),
    ("add", {"a": "1", "b": 2}), ("add", {"a": 1, "b": 2, "extra": 3}),
    ("weather", {"city": ""}), ("weather", {"city": 123}), ("weather", []),
])
def test_untrusted_calls_cannot_escape_the_allowlist(name, args):
    ledger = tools.SyntheticTools()
    with pytest.raises(ValueError):
        ledger.execute(name, args)
    assert ledger.calls == []


def test_retry_is_deterministic_and_state_is_per_conversation():
    a, b = tools.SyntheticTools(True), tools.SyntheticTools()
    assert a.execute("weather", {"city": "Kraków"})["retryable"]
    assert a.execute("weather", {"city": "Kraków"})["temperature_c"] == 21
    assert b.calls == []
    assert b.execute("add", {"a": 17, "b": 23}) == {"value": 40}
    assert a.counts() == {"weather": 2}
    assert b.counts() == {"add": 1}
