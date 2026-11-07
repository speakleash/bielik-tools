import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("transformers")
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("control_tokenization_test", ROOT / "scripts" / "check_control_tag_tokenization.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(["synthetic SAFE BEFORE AFTER"], trainers.BpeTrainer(
        vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    backend.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in probe.TAGS])
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize("family,file", [("v30", "bielik_advanced_chat_template.jinja"),
                                       ("v31", "bielik_v3.2_chat_template.jinja")])
def test_untrusted_control_tags_never_add_structure(tokenizer, family, file):
    rows = probe.check(tokenizer, (ROOT / "tools" / file).read_text(), family)
    assert len(rows) == 36
    assert all(row["passed"] for row in rows), [row for row in rows if not row["passed"]]


def test_disabled_sanitizer_is_detected(tokenizer):
    source = (ROOT / "tools" / "bielik_advanced_chat_template.jinja").read_text()
    source = '{% set sanitize_input = false %}' + source
    rows = probe.check(tokenizer, source, "v30")
    assert all(not row["passed"] for row in rows)


def test_missing_control_token_is_a_failure(tokenizer):
    source = (ROOT / "tools" / "bielik_advanced_chat_template.jinja").read_text()
    original = tokenizer.convert_tokens_to_ids
    tokenizer.convert_tokens_to_ids = lambda t: None if t == "<think>" else original(t)
    rows = probe.check(tokenizer, source, "v30")
    assert all(not row["passed"] for row in rows)


def test_normalizer_undoing_escaping_is_detected(tokenizer):
    source = (ROOT / "tools" / "bielik_advanced_chat_template.jinja").read_text()
    original = tokenizer.encode
    tokenizer.encode = lambda text, *a, **k: original(text.replace("\u200b", ""), *a, **k)
    rows = probe.check(tokenizer, source, "v30")
    assert all(not row["passed"] for row in rows)
    assert all(any("extra structural token ID" in e for e in row["errors"]) for row in rows)
    assert all(not any("not escaped" in e for e in row["errors"]) for row in rows)
