"""Tool parser for Bielik models (v3.0, v3.1, v3.2) on vLLM >= 0.19.0.

Compatible with ToolParser.__init__(tokenizer, tools=...) and sets
supports_required_and_named=False so tag-wrapped <tool_call> output uses
extract_tool_calls / adjust_request instead of vLLM's JSON required path.

For vLLM 0.15.0–0.18.x use bielik_vllm_tool_parser_v0.15.0.py.
"""
import json
import re
from typing import Union, Sequence


from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
try:
    from vllm.entrypoints.generate.base.protocol import (
        DeltaFunctionCall, DeltaMessage, DeltaToolCall,
        ExtractedToolCallInformation, FunctionCall, ToolCall,
    )
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("vllm.entrypoints.generate"):
        raise
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaFunctionCall, DeltaMessage, DeltaToolCall,
        ExtractedToolCallInformation, FunctionCall, ToolCall,
    )
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager
from vllm.utils import random_uuid

logger = init_logger(__name__)

# Bound whitespace around wrappers too: after reasoning, unconstrained \s*
# can otherwise consume the remaining output budget on newlines alone.
_REQUIRED_TOOL_REGEX = r"\s{0,2}(<tool_call>\s{0,2}\{[\s\S]*\}\s{0,2}</tool_call>\s{0,2})+"


# A named tool_choice forces exactly one call of that function. The prefix is
# shared by every named regex so streaming can recognise forced output.
_NAMED_TOOL_PREFIX = r"\s{0,2}<tool_call>\s{0,2}\{[ \t\r\n]{0,2}\"name\"[ \t\r\n]{0,2}:[ \t\r\n]{0,2}"


def _json_object_regex(depth: int = 3) -> str:
    """Regex for a JSON object with bounded nesting.

    Nesting is capped: the arguments object plus ``depth`` nested levels
    (4 levels in total by default); deeper arguments cannot be generated under
    a named tool_choice. A named call must be exactly one call, so the
    arguments cannot use ``[\\s\\S]*`` (it would swallow following calls).
    Strings accept JSON escapes and unescaped characters above U+001F,
    including ``<``.
    """
    ws = r"[ \t\r\n]{0,4}"
    string = r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
    number = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    value = f"(?:{string}|{number}|true|false|null)"
    for _ in range(depth):
        arr = rf"\[{ws}(?:{value}{ws}(?:,{ws}{value}{ws})*)?\]"
        obj = (rf"\{{{ws}(?:{string}{ws}:{ws}{value}{ws}"
               rf"(?:,{ws}{string}{ws}:{ws}{value}{ws})*)?\}}")
        value = f"(?:{string}|{number}|true|false|null|{arr}|{obj})"
    return (rf"\{{{ws}(?:{string}{ws}:{ws}{value}{ws}"
            rf"(?:,{ws}{string}{ws}:{ws}{value}{ws})*)?\}}")


_NAMED_MAX_DEPTH = 4  # arguments object + 3 nested levels (_json_object_regex)
_NAMED_ARGUMENTS_REGEX = _json_object_regex(_NAMED_MAX_DEPTH - 1)


def _schema_depth(schema) -> int:
    """Object/array nesting depth of a JSON schema (the root object counts 1).

    Best effort, used only for a warning: ``$ref``/``$defs`` and
    ``prefixItems`` are not followed.
    """
    if not isinstance(schema, dict):
        return 0
    depth = 0
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key) or []:
            depth = max(depth, _schema_depth(sub))
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "object" in kinds or "properties" in schema:
        children = list((schema.get("properties") or {}).values())
        if isinstance(schema.get("additionalProperties"), dict):
            children.append(schema["additionalProperties"])
        depth = max(depth, 1 + max((_schema_depth(c) for c in children), default=0))
    if "array" in kinds or "items" in schema:
        items = schema.get("items")
        items = items if isinstance(items, list) else [items]
        depth = max(depth, 1 + max((_schema_depth(i) for i in items), default=0))
    return depth


def _named_tool_regex(name: str) -> str:
    quoted = re.escape(json.dumps(name, ensure_ascii=False))
    return (_NAMED_TOOL_PREFIX + quoted
            + r"[ \t\r\n]{0,2},[ \t\r\n]{0,2}\"arguments\"[ \t\r\n]{0,2}:[ \t\r\n]{0,2}"
            + _NAMED_ARGUMENTS_REGEX + r"[ \t\r\n]{0,2}\}\s{0,2}</tool_call>\s{0,2}")


def _is_forced_regex(regex) -> bool:
    return isinstance(regex, str) and (
        regex == _REQUIRED_TOOL_REGEX or regex.startswith(_NAMED_TOOL_PREFIX))


def _tool_call_end(text: str, start: int) -> int:
    """Use the same quote-aware scanner for complete and incremental input."""
    return _StreamingBoundary().find(text, start)


def _checked_tool_call(payload: str) -> tuple[dict, str]:
    def invalid_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")
    call = json.loads(payload, parse_constant=invalid_constant)
    if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
            or not call["name"] or not isinstance(call.get("arguments"), dict)):
        raise ValueError("Expected a tool name and an arguments object")
    arguments = json.dumps(call["arguments"], ensure_ascii=False, allow_nan=False)
    return call, arguments


class _StreamingBoundary:
    """Scan each payload character once, retaining split wrapper prefixes."""
    def __init__(self):
        self.start = None
        self.offset = 0
        self.quoted = self.escaped = False

    def find(self, text: str, start: int) -> int:
        if self.start != start:
            self.start, self.offset = start, start
            self.quoted = self.escaped = False
        closing = "</tool_call>"
        while self.offset < len(text):
            index = self.offset
            char = text[index]
            if self.quoted:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.quoted = False
            elif char == '"':
                self.quoted = True
            elif char == "<":
                if text.startswith(closing, index):
                    return index
                if len(text) - index < len(closing) and closing.startswith(text[index:]):
                    return -1  # revisit only this short partial wrapper
            self.offset += 1
        return -1


# Registered as "bielik" so --tool-call-parser bielik works when this file is
# loaded via --tool-parser-plugin. Load only one Bielik tool parser file per
# process: the legacy per-version files register the same name.
@ToolParserManager.register_module("bielik")
class BielikToolParser(ToolParser):
    """Tool parser for Bielik chat templates (advanced 3.0 and v3.1/v3.2)."""

    # Tag-wrapped tool calls are incompatible with vLLM's default JSON
    # "required"/named guided path — use extract_tool_calls (+ adjust_request).
    supports_required_and_named: bool = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if isinstance(self.model_tokenizer, MistralTokenizer):
            logger.warning("Detected Mistral tokenizer when using a Bielik model")
            self.model_tokenizer = self.model_tokenizer.tokenizer

        # Streaming state: not every vLLM base class initializes these.
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self._stream_text_offset = 0
        self._stream_boundary = _StreamingBoundary()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Bielik Tool parser could not locate tool call start/end tokens in the tokenizer!")

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because Bielik uses the special tokens
            # to indicated the start and end of the tool calls information.
            request.skip_special_tokens = False
            if request.tool_choice == "required":
                # Bielik wraps tool calls in <tool_call>…</tool_call> tags
                # which is incompatible with vLLM's built-in "required"
                # handler (expects a raw JSON array).  Redirect to the
                # tag-aware tool-parser path ("auto") but enforce tool
                # generation via guided regex decoding.
                #
                # NOTE: adjust_request runs AFTER template rendering, so we
                # cannot inject template variables here.  For the strongest
                # enforcement pass tool_choice via chat_template_kwargs too:
                #   chat_template_kwargs={"tool_choice": "required"}
                # The grammar starts after reasoning has ended. Require the
                # call immediately, so a prose lead-in cannot consume the
                # output budget. Auto mode still allows ordinary text.
                # Keep arbitrary characters inside arguments (including '<').
                request.tool_choice = "auto"
                request.response_format = None
                request.structured_outputs = StructuredOutputsParams(
                    regex=_REQUIRED_TOOL_REGEX
                )
            elif getattr(getattr(request.tool_choice, "function", None), "name", None):
                # Named tool_choice: vLLM's JSON-based path does not fit the
                # tag-wrapped format, so without a constraint the model could
                # call other tools (and finish_reason ended up "stop").
                # Force exactly one call of the requested function.
                name = request.tool_choice.function.name
                for tool in request.tools:
                    function = getattr(tool, "function", None)
                    if getattr(function, "name", None) == name and \
                            _schema_depth(getattr(function, "parameters", None)) > _NAMED_MAX_DEPTH:
                        logger.warning(
                            "Named tool_choice %r: parameters nest deeper than %d levels; "
                            "the constrained output cannot contain such arguments.",
                            name, _NAMED_MAX_DEPTH)
                # finish_reason becomes "tool_calls" (vLLM and OpenAI report
                # "stop" for a forced function); clients usually check tool_calls.
                request.tool_choice = "auto"
                request.response_format = None
                request.structured_outputs = StructuredOutputsParams(
                    regex=_named_tool_regex(name)
                )
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        cursor, content, calls = 0, [], []
        while cursor < len(model_output):
            start = model_output.find(self.tool_call_start_token, cursor)
            if start < 0:
                content.append(model_output[cursor:])
                break
            content.append(model_output[cursor:start])
            payload_start = start + len(self.tool_call_start_token)
            end = _tool_call_end(model_output, payload_start)
            if end < 0:
                # Even valid JSON is not a complete call until its wrapper
                # closes. A length-limited response must not execute early.
                content.append(model_output[start:])
                break
            next_cursor = end + len(self.tool_call_end_token)
            try:
                call, arguments = _checked_tool_call(model_output[payload_start:end])
                calls.append(ToolCall(type="function", function=FunctionCall(
                    name=call["name"], arguments=arguments)))
            except (ValueError, TypeError, RecursionError):
                logger.warning("Invalid completed tool call in non-streaming output")
                content.append(model_output[start:next_cursor])
            cursor = next_cursor
        text = "".join(content)
        return ExtractedToolCallInformation(tools_called=bool(calls), tool_calls=calls,
                                           content=text if text else None)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        # current_text is cumulative. Consume each text range/closed call once;
        # never concatenate delta_text again or infer state from balanced counts.
        constraints = getattr(request, "structured_outputs", None)
        required_output = constraints is not None and _is_forced_regex(constraints.regex)
        # Guided decoding can spell the tags using ordinary tokens instead of
        # their registered special IDs. Required output is known to consist
        # only of calls; buffer partial text tags rather than leaking them.
        if not required_output and self.tool_call_start_token_id not in current_token_ids:
            self._stream_text_offset = len(current_text)
            return DeltaMessage(content=delta_text) if delta_text else None

        cursor = self._stream_text_offset
        content = []
        calls = []
        while cursor < len(current_text):
            start = current_text.find(self.tool_call_start_token, cursor)
            if start < 0:
                if required_output:
                    if not current_text[cursor:].strip():
                        cursor = len(current_text)
                    break
                content.append(current_text[cursor:])
                cursor = len(current_text)
                break
            if not required_output:
                content.append(current_text[cursor:start])
            payload_start = start + len(self.tool_call_start_token)
            end = self._stream_boundary.find(current_text, payload_start)
            if end < 0:
                cursor = start  # wait for the rest of this call; do not emit partial JSON
                break
            next_cursor = end + len(self.tool_call_end_token)
            try:
                call, arguments = _checked_tool_call(current_text[payload_start:end])
            except (ValueError, TypeError, RecursionError):
                # Preserve malformed completed blocks as text; never execute them.
                logger.warning("Invalid completed tool call in streaming output")
                content.append(current_text[start:next_cursor])
                cursor = next_cursor
                continue

            self.current_tool_id += 1
            self.current_tool_name_sent = True
            self.prev_tool_call_arr.append(call)
            self.streamed_args_for_tool.append(arguments)
            calls.append(DeltaToolCall(
                index=self.current_tool_id,
                type="function",
                id=f"chatcmpl-tool-{random_uuid()}",
                function=DeltaFunctionCall(name=call["name"], arguments=arguments),
            ))
            cursor = next_cursor

        self._stream_text_offset = cursor
        text = "".join(content)
        if not text and not calls:
            return None
        return DeltaMessage(content=text or None, tool_calls=calls)
