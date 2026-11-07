from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
try:
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
except ModuleNotFoundError as exc:
    if not exc.name.startswith("vllm.entrypoints.generate"):
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


@ReasoningParserManager.register_module("bielik")
class BielikReasoningParser(ReasoningParser):
    """
    Reasoning parser for Bielik models.

    Delegates to DeepSeekR1ReasoningParser when thinking is enabled
    (``enable_thinking=True`` in ``chat_template_kwargs``), and to
    IdentityReasoningParser otherwise — so that the entire model output
    is returned as content, not reasoning.
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
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))

        self._parser = None
        self._thinking = False
        if enable_thinking:
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
        if self._thinking and hasattr(request, "skip_special_tokens"):
            request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # vLLM also calls this on the prompt. The advanced template mentions
        # <think>...</think> in the system prompt, which must not look like
        # "reasoning already ended"; only inspect the current assistant turn.
        start = self.vocab.get("<|im_start|>")
        if start is not None:
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] == start:
                    input_ids = input_ids[i:]
                    break
        return self._parser.is_reasoning_end(input_ids)

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
