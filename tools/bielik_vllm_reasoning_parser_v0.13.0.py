from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                               DeltaMessage,
                                               ResponsesRequest)
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.logger import init_logger


logger = init_logger(__name__)
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

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))

        self._parser = None
        if enable_thinking:
            try:
                self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
            except RuntimeError:
                # Tokenizers without <think>/</think> special tokens (e.g.
                # Bielik-4.5B-v3.0) cannot be split by token id. Return the
                # whole output as content instead of failing the request.
                _warn_missing_think_tokens()
        if self._parser is None:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
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
