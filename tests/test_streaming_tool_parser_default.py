"""Parser-only tests: no model weights, server, or GPU execution required."""
import importlib.util
import inspect
import os
import json
from functools import lru_cache
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
try:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
except ModuleNotFoundError:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_plugin(kind):
    suffix = os.environ.get('BIELIK_TEST_TOOL_SUFFIX', '') if kind == 'tool' else '_v3.1'
    path = ROOT / 'tools' / f'bielik_vllm_{kind}_parser{suffix}.py'
    spec = importlib.util.spec_from_file_location(f'bielik_{kind}_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_plugin('tool').BielikToolParser


def make_tool(tokenizer, tools):
    if 'tools' in inspect.signature(TOOL.__init__).parameters:
        return TOOL(tokenizer, tools)
    return TOOL(tokenizer)


@pytest.fixture(scope='module')
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(['{"name":"calc","arguments":{"x":2}} hello world'] * 10,
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    backend.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in
                               ['<tool_call>', '</tool_call>', '<think>', '</think>']])
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.fixture
def request_data():
    return ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'test'}],
        tools=[{'type': 'function', 'function': {'name': 'calc', 'parameters': {'type': 'object'}}}],
        tool_choice='auto')


def collect(parser, tokenizer, request, text, batch):
    ids = tokenizer.encode(text, add_special_tokens=False)
    previous_text, previous_ids, deltas = '', [], []
    for pos in range(0, len(ids), batch):
        current_ids = ids[:pos + batch]
        current_text = tokenizer.decode(current_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert current_text.startswith(previous_text)
        delta = parser.extract_tool_calls_streaming(previous_text, current_text,
            current_text[len(previous_text):], previous_ids, current_ids, ids[pos:pos + batch], request)
        if delta:
            deltas.append(delta)
        previous_text, previous_ids = current_text, current_ids
    return deltas


def flatten(deltas):
    calls = {}
    for delta in deltas:
        for call in delta.tool_calls or []:
            row = calls.setdefault(call.index, {'arguments': ''})
            if call.id:
                assert 'id' not in row
                row['id'] = call.id
            if call.function and call.function.name:
                row['name'] = call.function.name
            if call.function and call.function.arguments:
                row['arguments'] += call.function.arguments
    return [(row['name'], json.loads(row['arguments'])) for _, row in sorted(calls.items())]


@pytest.mark.parametrize('arguments', [{}, {'x': 2}, {'x': -2.5, 'b': True, 'v': None},
    {'nested': {'items': [1, 'quoted "text"', 'Zażółć 🙂', '\\n']}}])
@pytest.mark.parametrize('batch', [1, 2, 3, 7, 17, 100000])
def test_complete_arguments_independent_of_chunking(tokenizer, request_data, arguments, batch):
    text = '<tool_call>' + json.dumps({'name': 'calc', 'arguments': arguments}, ensure_ascii=True) + '</tool_call>'
    parser = make_tool(tokenizer, request_data.tools)
    deltas = collect(parser, tokenizer, request_data, text, batch)
    assert flatten(deltas) == [('calc', arguments)]
    assert not ''.join(d.content or '' for d in deltas)
    assert len(parser.prev_tool_call_arr) == len(parser.streamed_args_for_tool)
    for call, streamed in zip(parser.prev_tool_call_arr, parser.streamed_args_for_tool):
        assert json.loads(streamed) == call['arguments']
    if hasattr(parser, 'get_remaining_unstreamed_args'):
        assert parser.get_remaining_unstreamed_args() == ''


@pytest.mark.parametrize('batch', [1, 3, 17, 100000])
def test_text_and_multiple_calls_are_never_lost_or_duplicated(tokenizer, request_data, batch):
    text = ('Before <tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>'
            '\nBetween <tool_call>{"name":"calc","arguments":{}}</tool_call> After')
    parser = make_tool(tokenizer, request_data.tools)
    deltas = collect(parser, tokenizer, request_data, text, batch)
    assert flatten(deltas) == [('calc', {'x': 1}), ('calc', {})]
    assert ''.join(d.content or '' for d in deltas) == 'Before \nBetween  After'
    assert len({c.id for d in deltas for c in d.tool_calls or []}) == 2
    assert len(parser.prev_tool_call_arr) == len(parser.streamed_args_for_tool)
    for call, streamed in zip(parser.prev_tool_call_arr, parser.streamed_args_for_tool):
        assert json.loads(streamed) == call['arguments']
    if hasattr(parser, 'get_remaining_unstreamed_args'):
        assert parser.get_remaining_unstreamed_args() == ''


@pytest.mark.parametrize('batch', [1, 3, 100000])
def test_invalid_call_is_text_not_executable(tokenizer, request_data, batch):
    text = "Before <tool_call>{'name':'calc','arguments':{}}</tool_call> After"
    deltas = collect(make_tool(tokenizer, request_data.tools), tokenizer, request_data, text, batch)
    assert flatten(deltas) == []
    assert ''.join(d.content or '' for d in deltas) == text


def test_incomplete_call_does_not_execute(tokenizer, request_data):
    deltas = collect(make_tool(tokenizer, request_data.tools), tokenizer, request_data,
                     '<tool_call>{"name":"calc","arguments":', 1)
    assert flatten(deltas) == []


@pytest.mark.parametrize('batch', [1, 3, 17, 100000])
def test_reasoning_quote_is_not_a_tool_call(tokenizer, request_data, batch):
    DelegatingParser = pytest.importorskip(
        'vllm.parser.abstract_parser',
        reason='Unified reasoning/tool parser is unavailable on older vLLM',
    ).DelegatingParser
    if not hasattr(DelegatingParser, 'parse_delta'):
        pytest.skip('This integration test requires the newer parse_delta API')
    reasoning = load_plugin('reasoning').BielikReasoningParser
    class Combined(DelegatingParser):
        reasoning_parser_cls = reasoning
        tool_parser_cls = TOOL
    parser = Combined(tokenizer, request_data.tools, chat_template_kwargs={'reasoning_effort': 'low'})
    # Older DelegatingParser leaves plugin construction to _WrappedParser.
    if parser.reasoning_parser is None:
        parser.reasoning_parser = reasoning(tokenizer, chat_template_kwargs={'reasoning_effort': 'low'})
    if parser.tool_parser is None:
        parser.tool_parser = make_tool(tokenizer, request_data.tools)
    text = ('Example <tool_call>{"name":"fake","arguments":{}}</tool_call> only.'
            '</think><tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>')
    ids = tokenizer.encode(text, add_special_tokens=False)
    previous, deltas = '', []
    prompt = tokenizer.encode('<think>', add_special_tokens=False)
    for pos in range(0, len(ids), batch):
        current = tokenizer.decode(ids[:pos + batch], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        options = {'prompt_token_ids': prompt}
        if 'finished' in inspect.signature(parser.parse_delta).parameters:
            options['finished'] = pos + batch >= len(ids)
        delta = parser.parse_delta(current[len(previous):], ids[pos:pos + batch], request_data,
                                   **options)
        if delta:
            deltas.append(delta)
        previous = current
    assert flatten(deltas) == [('calc', {'x': 2})]
    assert 'fake' in ''.join(d.reasoning or '' for d in deltas)


def test_required_calls_use_tag_aware_parser(tokenizer, request_data):
    request_data.tool_choice = 'required'
    request_data.skip_special_tokens = True
    parser = make_tool(tokenizer, request_data.tools)
    adjusted = parser.adjust_request(request_data)
    assert adjusted.skip_special_tokens is False
    assert adjusted.tool_choice == 'auto'
    assert adjusted.structured_outputs.regex
