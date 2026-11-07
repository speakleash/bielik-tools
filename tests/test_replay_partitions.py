"""Property-style partitions include every cut and reproducible random cuts."""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("vllm")
from test_parser_replay import make_capture, replay_module
from test_streaming_tool_parser_v31 import tokenizer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("replay_partitions_test", SCRIPTS / "check_replay_partitions.py")
parts = importlib.util.module_from_spec(spec)
sys.modules.setdefault("parser_replay", replay_module)
spec.loader.exec_module(parts)


@pytest.mark.parametrize("length", [0, 1, 2, 19, 1000])
def test_partition_covers_each_token_once_and_is_repeatable(length):
    for seed in range(20):
        boundaries = parts.partition(length, seed, 7)
        assert boundaries == parts.partition(length, seed, 7)
        assert (boundaries[-1] if boundaries else 0) == length
        assert all(0 < b - a <= 7 for a, b in zip([0, *boundaries], boundaries))


@pytest.mark.parametrize("text,thinking,tools", [
    ("Zażółć gęślą 😀 i dosłowny znak �.", False, False),
    ('<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>', False, True),
    ('<tool_call>{"name":"calc","arguments":{"text":"cytat: \\" i \\\\"}}</tool_call>', False, True),
    ('<tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>'
     '<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>', False, True),
    ('Przykład <tool_call>{"name":"fake","arguments":{}}</tool_call>.'
     '</think><tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>', True, True),
    ("Sprawdzam 😀.</think>Wynik: 391.", True, False),
    ("", False, False),
])
def test_all_single_cuts_and_random_partitions_match(tokenizer, text, thinking, tools):
    record = make_capture(tokenizer, text, thinking=thinking, tools=tools)
    expected = replay_module.replay(record, tokenizer)
    length = len(record["output_token_ids"])
    for cut in range(1, length):
        assert replay_module.replay(record, tokenizer, [cut, length]) == expected
    report = parts.check(record, tokenizer, seed=1701, trials=30, max_chunk=13)
    assert report["passed"], report


def test_known_literal_tag_parity_failure_is_reported(tokenizer):
    record = make_capture(tokenizer, '<tool_call>{"name":"calc","arguments":{}}</tool_call>', literal=True)
    report = parts.check(record, tokenizer, trials=3)
    assert report["passed"] is False
    assert all(not p["passed"] for p in report["partitions"])


@pytest.mark.parametrize("trials,max_chunk", [(0, 2), (-1, 2), (2, 0)])
def test_empty_or_invalid_test_runs_fail(tokenizer, trials, max_chunk):
    with pytest.raises(ValueError):
        parts.check(make_capture(tokenizer, "hello", tools=False), tokenizer,
                    trials=trials, max_chunk=max_chunk)


def test_partition_exception_is_saved_as_failure(tokenizer, monkeypatch):
    original = parts.replay
    def fail_stream(record, tok, boundaries=None):
        if boundaries is not None:
            raise ValueError("synthetic parser failure")
        return original(record, tok)
    monkeypatch.setattr(parts, "replay", fail_stream)
    report = parts.check(make_capture(tokenizer, "hello", tools=False), tokenizer, trials=1)
    assert not report["passed"]
    assert len(report["partitions"]) == 3
    assert all(p["error"] == "ValueError: synthetic parser failure" for p in report["partitions"])


def test_cli_reports_failure_and_never_overwrites(tokenizer, monkeypatch, tmp_path):
    import json
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    source, output = tmp_path / "capture.json", tmp_path / "report.json"
    source.write_text(json.dumps(make_capture(tokenizer,
        '<tool_call>{"name":"calc","arguments":{}}</tool_call>', literal=True)))
    argv = [str(source), "--tokenizer", "offline", "--trials", "1", "--output", str(output)]
    assert parts.main(argv) == 1
    assert json.loads(output.read_text())["passed"] is False
    with pytest.raises(FileExistsError):
        parts.main(argv)
