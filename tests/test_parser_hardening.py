"""Streaming-state initialisation and hook delegation of the Bielik parsers."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("vllm")

from test_streaming_tool_parser_v31 import TOOL, tokenizer  # noqa: E402,F401

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
VOCAB = {"<think>": 10, "</think>": 11, "a": 1}


class FakeTokenizer:
    def get_vocab(self):
        return VOCAB


def _load(filename):
    path = TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v31_tool_parser_initialises_streaming_state(tokenizer):
    parser = TOOL(tokenizer, None)
    assert parser.current_tool_id == -1
    assert parser.current_tool_name_sent is False
    assert parser.prev_tool_call_arr == []
    assert parser.streamed_args_for_tool == []


@pytest.mark.parametrize("filename", [
    "bielik_vllm_reasoning_parser.py"])
def test_reasoning_wrapper_delegates_prompt_and_tag_hooks(filename):
    module = _load(filename)
    parser = module.BielikReasoningParser(
        FakeTokenizer(), chat_template_kwargs={"enable_thinking": True})
    calls = []

    class Inner:
        def adjust_initial_state_from_prompt(self, ids):
            calls.append(("prompt", list(ids)))

        def prepare_structured_tag(self, tag, tool_server):
            calls.append(("tag", tag))
            return "changed"

    parser._parser = Inner()
    parser.adjust_initial_state_from_prompt([1, 2])
    assert parser.prepare_structured_tag("orig", None) == "changed"
    assert calls == [("prompt", [1, 2]), ("tag", "orig")]


@pytest.mark.parametrize("filename", [
    "bielik_vllm_reasoning_parser.py"])
def test_reasoning_wrapper_hooks_are_noops_without_inner_support(filename):
    module = _load(filename)
    parser = module.BielikReasoningParser(FakeTokenizer(), chat_template_kwargs={})
    parser._parser = object()
    parser.adjust_initial_state_from_prompt([1])
    assert parser.prepare_structured_tag("orig", None) == "orig"
