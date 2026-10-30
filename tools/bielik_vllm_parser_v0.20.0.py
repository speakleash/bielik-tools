"""Native unified Bielik parser for unmodified vLLM 0.20.0.

Pass this same file to --tool-parser-plugin and --reasoning-parser-plugin,
and select bielik_020 for both parser names. The reasoning-plugin flag also
loads the registration in engine workers.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import sys

if importlib.metadata.version("vllm").split("+")[0] != "0.20.0":
    raise RuntimeError(
        "This plugin targets vLLM 0.20.0; use the parser for your version"
    )

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager


def _load_sibling(filename, module_name):
    path = Path(__file__).resolve().with_name(filename)
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError(
                f"Module name already used for another file: {module_name}"
            )
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


ToolParser = _load_sibling(
    "bielik_vllm_tool_parser.py", "_bielik_020_tool"
).BielikToolParser
ReasoningParser = _load_sibling(
    "bielik_vllm_reasoning_parser_v3.1.py", "_bielik_020_reasoning"
).BielikReasoningParser


class Bielik020Parser(DelegatingParser):
    """Retain reasoning across the legacy DelegatingParser's tool handoff.

    The native callback hooks let the plugin preserve the same delta without
    replacing vLLM methods or changing installed package files. All transient
    handoff data belongs to this request's parser instance.
    """

    reasoning_parser_cls = ReasoningParser
    tool_parser_cls = ToolParser

    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer)
        self.reasoning_parser = self.reasoning_parser_cls(tokenizer, **kwargs)
        self.tool_parser = self.tool_parser_cls(tokenizer, tools)
        self._collect_handoff = False
        self._handoff_reasoning = None

    def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None):
        self._collect_handoff = True
        self._handoff_reasoning = None
        try:
            return super().parse_delta(
                delta_text, delta_token_ids, request, prompt_token_ids=prompt_token_ids
            )
        finally:
            self._collect_handoff = False
            self._handoff_reasoning = None

    def extract_reasoning_streaming(self, *args, **kwargs):
        delta = super().extract_reasoning_streaming(*args, **kwargs)
        if self._collect_handoff and delta is not None:
            self._handoff_reasoning = delta.reasoning
        return delta

    def extract_tool_calls_streaming(self, *args, **kwargs):
        delta = super().extract_tool_calls_streaming(*args, **kwargs)
        reasoning = self._handoff_reasoning if self._collect_handoff else None
        if not reasoning:
            return delta
        if delta is None:
            return DeltaMessage(reasoning=reasoning)
        return delta.model_copy(
            update={"reasoning": reasoning + (delta.reasoning or "")}
        )


# Eager registration supports arbitrary plugin filenames and avoids relying on
# importable package names for a file loaded through --tool-parser-plugin.
ToolParserManager.register_module("bielik_020", module=ToolParser)
ReasoningParserManager.register_module("bielik_020", module=ReasoningParser)
ParserManager.register_module("bielik_020", module=Bielik020Parser)
