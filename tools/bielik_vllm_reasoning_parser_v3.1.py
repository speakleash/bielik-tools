from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
try:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
except ModuleNotFoundError:
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


# Registered as "bielik" so --reasoning-parser bielik works when this file is
# loaded via --reasoning-parser-plugin. Do not load this plugin and the legacy
# bielik_vllm_reasoning_parser.py in the same process (same registration name).
@ReasoningParserManager.register_module("bielik")
class BielikReasoningParser(ReasoningParser):
    """
    Reasoning parser for Bielik models with the v3.1 chat template.

    Delegates to DeepSeekR1ReasoningParser when thinking is enabled, and to
    IdentityReasoningParser otherwise — so that the entire model output is
    returned as content, not reasoning.

    Thinking is enabled when ``chat_template_kwargs`` has
    ``reasoning_effort`` set to anything other than ``"none"``. If
    ``reasoning_effort`` is absent, falls back to the legacy
    ``enable_thinking=True`` flag (advanced template).
    """

    @property
    def reasoning_start_str(self) -> str:
        return "<think>"

    @property
    def reasoning_end_str(self) -> str:
        return "</think>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        effort = chat_kwargs.get("reasoning_effort")
        # True when thinking was inferred only from the legacy flag. The v3.1
        # template ignores enable_thinking and forces an empty <think></think>
        # in the prompt, so the whole output is content. vLLM calls
        # is_reasoning_end(prompt_ids) once per request (streaming or not); the
        # result tells us whether the prompt already closed the think block.
        # The advanced template leaves it open, so nothing changes there.
        self._legacy_flag = False
        self._prompt_closed = None
        if effort is not None:
            enable_thinking = str(effort).lower() != "none"
        else:
            enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
            self._legacy_flag = enable_thinking

        self._parser = None
        if enable_thinking:
            if (self.vocab.get("<think>") is None
                    or self.vocab.get("</think>") is None):
                # Only missing delimiters trigger the content fallback. Other
                # initialization errors must surface instead of being hidden.
                _warn_missing_think_tokens()
            else:
                self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        if self._parser is None:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        result = self._parser.is_reasoning_end(input_ids)
        if self._legacy_flag and self._prompt_closed is None:
            self._prompt_closed = result
        return result

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        if self._legacy_flag and self._prompt_closed:
            # Same as streaming, where the closed prompt makes vLLM emit content.
            return None, model_output
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
