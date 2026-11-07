"""Reasoning parser for Bielik models (v3.0, v3.1, v3.2) on vLLM >= 0.19.0.

One parser for every Bielik chat template:

* v3.1/v3.2 templates switch thinking with ``reasoning_effort``
  (``none`` disables it);
* the advanced v3.0 template switches it with ``enable_thinking``.

``reasoning_effort`` wins when present (vLLM 0.32 also sets ``enable_thinking``
for a top-level ``reasoning_effort``). When thinking is requested but the
current assistant turn of the prompt already closes ``<think></think>`` (for
example ``enable_thinking=true`` sent to a v3.1 template, which ignores that
flag), the output is returned as content.
"""
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
try:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("vllm.entrypoints.generate"):
        raise
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.logger import init_logger


# Log under the "vllm" namespace: plugins are loaded by file path, so
# __name__ falls outside vLLM's configured logger hierarchy.
logger = init_logger("vllm.plugins.bielik_reasoning_parser")
_missing_think_tokens_warned = False


def _warn_missing_think_tokens() -> None:
    global _missing_think_tokens_warned
    if not _missing_think_tokens_warned:
        _missing_think_tokens_warned = True
        logger.warning(
            "Bielik reasoning parser: the tokenizer has no <think>/</think> "
            "special tokens; thinking output is returned as content.")


def thinking_requested(chat_template_kwargs) -> bool:
    """Whether the request asks for thinking (v3.1 or advanced template)."""
    kwargs = chat_template_kwargs or {}
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        return str(effort).strip().lower() != "none"
    return bool(kwargs.get("enable_thinking", False))


# Registered as "bielik". Load only one Bielik reasoning parser file per process
# (the legacy per-version file registers the same name).
@ReasoningParserManager.register_module("bielik")
class BielikReasoningParser(ReasoningParser):
    """
    Reasoning parser for Bielik models.

    Delegates to DeepSeekR1ReasoningParser when thinking is requested and to
    IdentityReasoningParser otherwise, so that the entire model output is
    returned as content.
    """

    @property
    def reasoning_start_str(self) -> str:
        return "<think>"

    @property
    def reasoning_end_str(self) -> str:
        return "</think>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._init_args = (tokenizer, args, kwargs)
        self._turn_start_id = self.vocab.get("<|im_start|>")

        self._parser = None
        self._thinking = False
        if thinking_requested(kwargs.get("chat_template_kwargs")):
            if (self.vocab.get("<think>") is None
                    or self.vocab.get("</think>") is None):
                # Only missing delimiters trigger the content fallback. Other
                # initialization errors must surface instead of being hidden.
                _warn_missing_think_tokens()
            else:
                self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
                self._thinking = True
        if self._parser is None:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # Some tokenizers (e.g. Bielik v3.0) flag <think>/</think> as special
        # tokens. With the default skip_special_tokens=True they are dropped
        # from the text and reasoning cannot be separated from the answer.
        # This overrides an explicit skip_special_tokens=True (same trade-off
        # as the Bielik tool parser).
        if self._thinking and hasattr(request, "skip_special_tokens"):
            request.skip_special_tokens = False
        return request

    def _current_turn(self, input_ids: Sequence[int]) -> tuple[list[int], bool]:
        """Token IDs from the last <|im_start|>, and whether one was found."""
        if self._turn_start_id is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == self._turn_start_id:
                    return list(input_ids[i:]), True
        return list(input_ids), False

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # vLLM calls this with the rendered prompt. Only the current assistant
        # turn counts: system prompts and earlier turns may quote
        # <think>...</think> (the advanced template does), which must not look
        # like "reasoning already ended".
        turn, is_prompt = self._current_turn(input_ids)
        ended = self._parser.is_reasoning_end(turn)
        if ended and is_prompt and self._thinking:
            # Thinking was requested, but the template already closed the
            # reasoning block for this turn, so the output is the answer.
            # Model output never contains <|im_start|>: this sees prompts only.
            # vLLM 0.32 calls this before parsing a non-streaming response;
            # extra choices (n > 1) get fresh parsers that are not told.
            self._parser = IdentityReasoningParser(
                self._init_args[0], *self._init_args[1], **self._init_args[2])
            self._thinking = False
        return ended

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def adjust_initial_state_from_prompt(self, prompt_token_ids) -> None:
        hook = getattr(self._parser, "adjust_initial_state_from_prompt", None)
        if hook is not None:
            hook(prompt_token_ids)

    def prepare_structured_tag(self, original_tag, tool_server):
        hook = getattr(self._parser, "prepare_structured_tag", None)
        if hook is None:
            return original_tag
        return hook(original_tag, tool_server)
