"""Replay the same token IDs; never execute a tool or call a server."""
import importlib.util
import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("vllm")
from test_streaming_tool_parser_v31 import tokenizer

PATH = Path(__file__).resolve().parents[1] / "scripts" / "parser_replay.py"
spec = importlib.util.spec_from_file_location("parser_replay_test", PATH)
replay_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_module)


def make_capture(tokenizer, text, *, thinking=False, tools=True, literal=False, choice="auto"):
    request = {"model": "offline", "messages": [{"role": "user", "content": "synthetic"}],
               "tool_choice": choice, "chat_template_kwargs": {
                   "reasoning_effort": "medium" if thinking else "none"}}
    if tools:
        request["tools"] = [{"type": "function", "function": {"name": "calc",
                            "parameters": {"type": "object"}}}]
    if not tools:
        request.pop("tool_choice")
    return {"format_version": 1, "request": request,
            "tokenizer_sha256": replay_module.fingerprint(tokenizer),
            "prompt_token_ids": tokenizer.encode("<think>" if thinking else "<think></think>",
                                                 add_special_tokens=False),
            "output_token_ids": tokenizer.encode(text, add_special_tokens=False,
                                                   split_special_tokens=literal),
            "output_text": text}


@pytest.mark.parametrize("text,thinking,tools", [
    ("Wynik to 42.", False, False),
    ('<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>', False, True),
    ('Obliczam.</think><tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>', True, True),
    ("Obliczam.</think>Wynik to 42.", True, False),
    ("", False, False),
])
def test_replay_parity_and_json_roundtrip(tokenizer, text, thinking, tools):
    record = json.loads(json.dumps(make_capture(tokenizer, text, thinking=thinking, tools=tools)))
    expected = replay_module.replay(record, tokenizer)
    assert replay_module.replay(record, tokenizer, list(range(1, len(record["output_token_ids"]) + 1))) == expected
    assert expected["tool_calls"] == ([{"name": "calc", "arguments": {"x": 2}}] if tools else [])
    assert bool(expected["reasoning"]) == thinking


def test_replay_preserves_literal_vs_dedicated_token_ids(tokenizer):
    text = '<tool_call>{"name":"calc","arguments":{}}</tool_call>'
    record = make_capture(tokenizer, text, literal=True)
    assert replay_module.replay(record, tokenizer)["tool_calls"]
    streamed = replay_module.replay(record, tokenizer, [len(record["output_token_ids"])])
    # This documented asymmetry must remain visible, not be hidden by encoding
    # captured text again and replacing ordinary IDs with dedicated tag IDs.
    assert not streamed["tool_calls"]
    assert streamed["content"] == text


def test_replay_rejects_a_different_tokenizer_or_corrupt_text(tokenizer):
    record = make_capture(tokenizer, "hello", tools=False)
    with pytest.raises(ValueError, match="fingerprint"):
        replay_module.replay({**record, "tokenizer_sha256": "wrong"}, tokenizer)
    with pytest.raises(ValueError, match="does not match"):
        replay_module.replay({**record, "output_text": "changed"}, tokenizer)


@pytest.mark.parametrize("boundaries", [[], [0], [1, 1], [999], [-1, 2]])
def test_invalid_partitions_fail(tokenizer, boundaries):
    with pytest.raises(ValueError, match="Boundaries"):
        replay_module.replay(make_capture(tokenizer, "hello", tools=False), tokenizer, boundaries)


@pytest.mark.parametrize("terminal_stop", [False, True])
def test_capture_preserves_raw_ids_and_never_executes_tools(tokenizer, monkeypatch, terminal_stop):
    from io import BytesIO

    text = '<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>'
    ids = tokenizer.encode(text, add_special_tokens=False)
    if terminal_stop:
        # vLLM changes finish_reason from stop to tool_calls after extraction.
        # Keep the underlying stop ID, but suppress its detokenized text.
        # Recent transformers exposes unset special tokens dynamically;
        # monkeypatch undo cannot delete that virtual attribute. Keep the
        # shared fixture untouched and configure a private tokenizer copy.
        tokenizer = copy.deepcopy(tokenizer)
        tokenizer.eos_token = "</think>"
    raw_ids = ids + ([tokenizer.eos_token_id] if terminal_stop else [])
    raw = {"prompt_token_ids": tokenizer.encode("user", add_special_tokens=False),
           "choices": [{"token_ids": raw_ids, "finish_reason": "tool_calls",
                        "message": {"tool_calls": []}}]}
    calls = []
    def post(request, timeout):
        calls.append(json.loads(request.data))
        return BytesIO(json.dumps(raw).encode())
    monkeypatch.setattr(replay_module.urllib.request, "urlopen", post)
    result = replay_module.capture("http://localhost:8000/v1", {"model": "offline"}, tokenizer)
    assert result["output_token_ids"] == raw_ids
    assert result["terminal_stop_suppressed"] == terminal_stop
    assert result["output_text"] == text
    assert calls == [{"model": "offline", "stream": False, "return_token_ids": True,
                      "skip_special_tokens": False}]


@pytest.mark.parametrize("payload", [{"stream": True}, {"n": 2}])
def test_capture_rejects_unsupported_response_shapes(tokenizer, payload):
    with pytest.raises(ValueError, match="requires"):
        replay_module.capture("http://localhost:1", payload, tokenizer)


@pytest.mark.parametrize("option", [
    {"stop": "STOP"}, {"stop_token_ids": [123]},
    {"include_stop_str_in_output": True}, {"ignore_eos": True},
])
def test_custom_stopping_is_rejected_before_network(tokenizer, option):
    with pytest.raises(ValueError, match="not supported"):
        replay_module.capture("http://localhost:1", option, tokenizer)


def test_prompt_primed_strip_decoder_preserves_leading_space():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"prompt": 0, "▁Hello": 1}, unk_token="prompt"))
    backend.decoder = decoders.Sequence([
        decoders.Replace("▁", " "), decoders.Fuse(), decoders.Strip(" ", left=1, right=0)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    assert tok.decode([1]) == "Hello"
    assert replay_module.decoded_prefixes(tok, [0], [1]) == [" Hello"]


@pytest.mark.parametrize("effort,explicit", [(None, {}), ("none", {}), ("medium", {}),
    ("medium", {"enable_thinking": False}), ("none", {"reasoning_effort": "medium", "enable_thinking": True})])
def test_replay_uses_native_request_template_mapping(tokenizer, monkeypatch, effort, explicit):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    record = make_capture(tokenizer, "hello", tools=False)
    record["request"]["chat_template_kwargs"] = explicit
    if effort is not None:
        record["request"]["reasoning_effort"] = effort
    seen = []
    original = ChatCompletionRequest.build_chat_params
    def mapping(self, *args, **kwargs):
        params = original(self, *args, **kwargs)
        seen.append(params.chat_template_kwargs)
        return params
    monkeypatch.setattr(ChatCompletionRequest, "build_chat_params", mapping)
    parser, request = replay_module.parser_for(record, tokenizer)
    assert len(seen) == 1
    expected = original(request, None, "auto").chat_template_kwargs
    assert seen[0] == expected
    if "enable_thinking" in explicit:
        assert seen[0]["enable_thinking"] is explicit["enable_thinking"]


def test_cli_saves_invalid_parser_arguments_as_failure(tokenizer, monkeypatch, tmp_path):
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    source, output = tmp_path / "capture.json", tmp_path / "result.json"
    source.write_text(json.dumps(make_capture(tokenizer, "hello", tools=False)))
    def malformed(*args):
        from types import SimpleNamespace
        return replay_module.normalized("", "", [SimpleNamespace(name="calc", arguments="invalid JSON")])
    monkeypatch.setattr(replay_module, "replay", malformed)
    assert replay_module.main(["replay", "--tokenizer", "offline", "--input", str(source),
                               "--output", str(output)]) == 1
    saved = json.loads(output.read_text())
    assert not saved["parity"]
    assert "JSONDecodeError" in saved["non_stream"]["error"]
    assert "JSONDecodeError" in saved["stream"]["error"]
