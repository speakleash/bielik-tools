"""Run with pytest in a vLLM environment; no server/model weights required."""
import ast
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('kind', ['tool', 'reasoning'])
@pytest.mark.parametrize('layout', ['new', 'old', 'broken_dependency'])
def test_protocol_import_fallback(kind, layout):
    """Exercise both layouts and ensure unrelated missing dependencies surface."""
    source = ROOT / 'tools' / f'bielik_vllm_{kind}_parser.py'
    tree = ast.parse(source.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.Try))
    imports = []
    marker = object()

    def fake_import(name, *args, **kwargs):
        imports.append(name)
        if name == 'vllm.entrypoints.generate.base.protocol':
            if layout == 'old':
                raise ModuleNotFoundError(name='vllm.entrypoints.generate')
            if layout == 'broken_dependency':
                raise ModuleNotFoundError(name='missing_dependency')
        return SimpleNamespace(**{key: marker for key in [
            'DeltaFunctionCall', 'DeltaMessage', 'DeltaToolCall',
            'ExtractedToolCallInformation', 'FunctionCall', 'ToolCall']})

    namespace = {'__builtins__': dict(vars(builtins), __import__=fake_import)}
    code = compile(ast.Module(body=[block], type_ignores=[]), str(source), 'exec')
    if layout == 'broken_dependency':
        with pytest.raises(ModuleNotFoundError):
            exec(code, namespace)
        assert len(imports) == 1
    else:
        exec(code, namespace)
        assert namespace['DeltaMessage'] is marker
        assert len(imports) == (2 if layout == 'old' else 1)


@pytest.mark.parametrize('kind', ['tool', 'reasoning'])
def test_real_vllm_plugin(kind):
    from tokenizers import AddedToken, Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    backend = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    backend.add_special_tokens([AddedToken(t, special=True) for t in
                               ['<tool_call>', '</tool_call>', '<think>', '</think>']])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    path = ROOT / 'tools' / f'bielik_vllm_{kind}_parser.py'
    spec = importlib.util.spec_from_file_location(f'compat_{kind}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'test'}])
    if kind == 'tool':
        parser = module.BielikToolParser(tokenizer, [])
        result = parser.extract_tool_calls('<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>', request)
        assert result.tools_called
        assert result.tool_calls[0].function.name == 'calc'
        assert json.loads(result.tool_calls[0].function.arguments) == {'x': 2}
    else:
        for thinking in [False, True]:
            parser = module.BielikReasoningParser(tokenizer, chat_template_kwargs={'enable_thinking': thinking})
            text = '<think>analysis</think>answer'
            reasoning, content = parser.extract_reasoning(text, request)
            assert (reasoning, content) == (('analysis', 'answer') if thinking else (None, text))
