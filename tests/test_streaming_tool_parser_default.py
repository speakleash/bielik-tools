"""Parser-only tests: no model weights, server, or GPU execution required."""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_plugin(kind):
    suffix = '' if kind == 'tool' else '_v3.1'
    path = ROOT / 'tools' / f'bielik_vllm_{kind}_parser{suffix}.py'
    spec = importlib.util.spec_from_file_location(f'bielik_{kind}_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_plugin('tool').BielikToolParser


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
    parser = TOOL(tokenizer, request_data.tools)
    deltas = collect(parser, tokenizer, request_data, text, batch)
    assert flatten(deltas) == [('calc', arguments)]
    assert not ''.join(d.content or '' for d in deltas)
    assert parser.get_remaining_unstreamed_args() == ''


@pytest.mark.parametrize('batch', [1, 3, 17, 100000])
def test_text_and_multiple_calls_are_never_lost_or_duplicated(tokenizer, request_data, batch):
    text = ('Before <tool_call>{"name":"calc","arguments":{"x":1}}</tool_call>'
            '\nBetween <tool_call>{"name":"calc","arguments":{}}</tool_call> After')
    parser = TOOL(tokenizer, request_data.tools)
    deltas = collect(parser, tokenizer, request_data, text, batch)
    assert flatten(deltas) == [('calc', {'x': 1}), ('calc', {})]
    assert ''.join(d.content or '' for d in deltas) == 'Before \nBetween  After'
    assert len({c.id for d in deltas for c in d.tool_calls or []}) == 2
    assert parser.get_remaining_unstreamed_args() == ''


@pytest.mark.parametrize('batch', [1, 3, 100000])
def test_invalid_call_is_text_not_executable(tokenizer, request_data, batch):
    text = "Before <tool_call>{'name':'calc','arguments':{}}</tool_call> After"
    deltas = collect(TOOL(tokenizer, request_data.tools), tokenizer, request_data, text, batch)
    assert flatten(deltas) == []
    assert ''.join(d.content or '' for d in deltas) == text


def test_incomplete_call_does_not_execute(tokenizer, request_data):
    deltas = collect(TOOL(tokenizer, request_data.tools), tokenizer, request_data,
                     '<tool_call>{"name":"calc","arguments":', 1)
    assert flatten(deltas) == []


@pytest.mark.parametrize('batch', [1, 3, 17, 100000])
def test_reasoning_quote_is_not_a_tool_call(tokenizer, request_data, batch):
    from vllm.parser.abstract_parser import DelegatingParser
    reasoning = load_plugin('reasoning').BielikReasoningParser
    class Combined(DelegatingParser):
        reasoning_parser_cls = reasoning
        tool_parser_cls = TOOL
    parser = Combined(tokenizer, request_data.tools, chat_template_kwargs={'reasoning_effort': 'low'})
    text = ('Example <tool_call>{"name":"fake","arguments":{}}</tool_call> only.'
            '</think><tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>')
    ids = tokenizer.encode(text, add_special_tokens=False)
    previous, deltas = '', []
    prompt = tokenizer.encode('<think>', add_special_tokens=False)
    for pos in range(0, len(ids), batch):
        current = tokenizer.decode(ids[:pos + batch], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        delta = parser.parse_delta(current[len(previous):], ids[pos:pos + batch], request_data,
                                   prompt_token_ids=prompt, finished=pos + batch >= len(ids))
        if delta:
            deltas.append(delta)
        previous = current
    assert flatten(deltas) == [('calc', {'x': 2})]
    assert 'fake' in ''.join(d.reasoning or '' for d in deltas)
