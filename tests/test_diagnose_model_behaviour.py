"""Offline tests for scripts/diagnose_model_behaviour.py (no server, no model files)."""
import importlib.util
import urllib.error
from pathlib import Path

import pytest

pytest.importorskip("transformers")

PATH = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_model_behaviour.py"
spec = importlib.util.spec_from_file_location("diagnose_model_behaviour", PATH)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

TAG_IDS = {"<think>": 32007, "</think>": 32008, "<tool_call>": 32005, "</tool_call>": 32006,
           "<|im_start|>": 32000, "<|im_end|>": 32001, "</s>": 2}


class FakeTokenizer:
    """Tags have dedicated IDs; every other character is its own token."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return TAG_IDS.get(token, self.unk_token_id)

    def decode(self, ids):
        names = {v: k for k, v in TAG_IDS.items()}
        return "".join(names.get(i) or chr(i - 1000) for i in ids)

    def encode(self, text, literal=False):
        """Encode text; dedicated IDs for tags unless literal=True."""
        ids, i = [], 0
        while i < len(text):
            tag = next((t for t in TAG_IDS if not literal and text.startswith(t, i)), None)
            if tag:
                ids.append(TAG_IDS[tag])
                i += len(tag)
            else:
                ids.append(ord(text[i]) + 1000)
                i += 1
        return ids


TOK = FakeTokenizer()
PROMPT_CLOSED = TOK.encode("<|im_start|>assistant\n<think></think>")
PROMPT_OPEN = TOK.encode("<|im_start|>assistant\n<think>")
CALL_EXP = {"think": False, "call": True, "names": {"get_weather"}}


def flags(output, prompt=PROMPT_CLOSED, exp=CALL_EXP, literal=False):
    return diag.Diagnoser(TOK).analyse(TOK.encode(output, literal), prompt, "stop", exp, diag.TOOLS)["flags"]


def test_clean_call():
    assert flags('<tool_call>{"name": "get_weather", "arguments": {"city": "Gdańsk"}}</tool_call>') == []


@pytest.mark.parametrize("arguments", ["[]", "null", "2", '"x"', '"[1]"'])
def test_arguments_that_are_not_an_object_are_flagged(arguments):
    out = '<tool_call>{"name": "get_weather", "arguments": %s}</tool_call>' % arguments
    assert "call_arguments_not_object" in flags(out)


def test_arguments_as_json_string_object():
    out = '<tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"G\\"}"}</tool_call>'
    found = flags(out)
    assert "call_arguments_as_string" in found and "call_arguments_not_object" not in found


def test_literal_tags_and_reasoning_structure():
    literal = flags('<tool_call>{"name": "get_weather", "arguments": {"city": "G"}}</tool_call>', literal=True)
    assert "literal:<tool_call>" in literal and "call_missing" in literal
    reopened = flags("<think>a</think>b", prompt=PROMPT_OPEN, exp={"think": True, "call": False})
    assert "think_reopened" in reopened


def run_main(monkeypatch, argv, post=None):
    monkeypatch.setattr(diag.AutoTokenizer, "from_pretrained", lambda *_: TOK)
    if post is not None:
        monkeypatch.setattr(diag, "post", post)
    monkeypatch.setattr("sys.argv", ["diagnose", "--model", "m", "--tokenizer", "t", *argv])
    return diag.main()


def unreachable(url, body):
    raise urllib.error.URLError("connection refused")


def test_unreachable_server_fails_even_without_strict(monkeypatch):
    assert run_main(monkeypatch, ["--only", "think_on_math"], unreachable) == 2
    assert run_main(monkeypatch, ["--only", "think_on_math", "--strict"], unreachable) == 2


def test_unknown_scenario_fails(monkeypatch):
    assert run_main(monkeypatch, ["--only", "no_such_case"], unreachable) == 2


def fake_server(output):
    def post(url, body):
        ids = TOK.encode(output)
        return {"prompt_token_ids": PROMPT_CLOSED,
                "choices": [{"token_ids": ids, "finish_reason": "stop"}] * body["n"]}
    return post


def test_exit_codes_for_clean_and_issue_runs(monkeypatch):
    clean = fake_server("Średnia prędkość to 80 km/h.<|im_end|>")
    assert run_main(monkeypatch, ["--only", "think_off_math", "--strict"], clean) == 0
    bad = fake_server('<tool_call>{"name": "x", "arguments": []}</tool_call><|im_end|>')
    assert run_main(monkeypatch, ["--only", "think_off_math"], bad) == 0
    assert run_main(monkeypatch, ["--only", "think_off_math", "--strict"], bad) == 1
