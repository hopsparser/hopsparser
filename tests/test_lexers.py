import json
import math
import pathlib
import tempfile
from typing import List, Literal, Tuple

import pytest
import torch
import transformers
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest_lazy_fixtures import lf as lazy_fixture

from hopsparser import lexers


# TODO: test the `from_config` methods


@given(
    char_embeddings_dim=st.integers(min_value=1, max_value=512),
    chars=st.lists(st.characters(), min_size=1),
    half_output_dim=st.integers(min_value=1, max_value=256),
    special_tokens=st.lists(st.text(min_size=2), max_size=8),
    test_text=st.lists(st.text(min_size=1), min_size=1),
)
def test_char_rnn_create_save_load(
    char_embeddings_dim: int,
    chars: list[str],
    half_output_dim: int,
    special_tokens: list[str],
    test_text: list[str],
):
    char_lexer = lexers.CharRNNLexer.from_chars(
        char_embeddings_dim=char_embeddings_dim,
        chars=chars,
        output_dim=2 * half_output_dim,
        special_tokens=special_tokens,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        char_lexer.save(tmp_path, save_weights=True)
        reloaded = lexers.CharRNNLexer.load(tmp_path)
    assert char_lexer.vocabulary == reloaded.vocabulary
    char_lexer.eval()
    reloaded.eval()
    with torch.inference_mode():
        orig_encoding = char_lexer(char_lexer.make_batch([char_lexer.encode(test_text)]))
        reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
    assert torch.equal(orig_encoding, reloaded_encoding)


@pytest.fixture
def remote_fasttext_model() -> str:
    return "lgrobol/fasttext-minuscule"


@pytest.fixture
def local_fasttext_model(
    test_data_dir: pathlib.Path,
) -> pathlib.Path:
    return test_data_dir / "fasttext_model.bin"


# TODO: do we really need lazy fixture here?
@pytest.fixture(
    params=[
        lazy_fixture("local_fasttext_model"),
        lazy_fixture("remote_fasttext_model"),
    ],
)
def fasttext_model(
    request,
) -> str | pathlib.Path:
    return request.param


# NOTE: the function-scoped fixture are only model paths/identifiers so it's ok.
@settings(deadline=1000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    special_tokens=st.lists(st.text(min_size=2), max_size=8),
    test_text=st.lists(st.text(min_size=1), min_size=1),
)
def test_fasttext_train_create_save_load(
    fasttext_model: str | pathlib.Path, special_tokens: list[str], test_text: list[str]
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        fasttext_lexer = lexers.FastTextLexer.from_fasttext_model(
            fasttext_model,
            special_tokens=special_tokens,
        )
        fasttext_lexer.save(tmp_path, save_weights=True)
        reloaded = lexers.FastTextLexer.load(tmp_path)
    assert fasttext_lexer.special_tokens == reloaded.special_tokens
    fasttext_lexer.eval()
    reloaded.eval()
    with torch.inference_mode():
        orig_encoding = fasttext_lexer(
            fasttext_lexer.make_batch([fasttext_lexer.encode(test_text)])
        )
        reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
    assert torch.equal(orig_encoding, reloaded_encoding)


@given(
    embeddings_dim=st.integers(min_value=1, max_value=512),
    words=st.lists(st.text(), min_size=1),
    word_dropout=st.floats(min_value=0.0, max_value=1.0),
    test_text=st.lists(st.text(min_size=1), min_size=1),
)
def test_word_embeddings_create_save_load(
    embeddings_dim: int,
    test_text: list[str],
    word_dropout: float,
    words: list[str],
):
    lexer = lexers.WordEmbeddingsLexer.from_words(
        embeddings_dim=embeddings_dim,
        word_dropout=word_dropout,
        words=words,
        unk_word=words[0],
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        lexer.save(tmp_path, save_weights=True)
        reloaded = lexers.WordEmbeddingsLexer.load(tmp_path)
    assert lexer.vocabulary == reloaded.vocabulary
    lexer.eval()
    reloaded.eval()
    with torch.inference_mode():
        orig_encoding = lexer(lexer.make_batch([lexer.encode(test_text)]))
        reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
    assert torch.equal(orig_encoding, reloaded_encoding)


@pytest.fixture(params=["lgrobol/flaubert-minuscule"])
def remote_transformer_model(
    request,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    model = transformers.AutoModel.from_pretrained(request.param)
    tokenizer = transformers.AutoTokenizer.from_pretrained(request.param, use_fast=True)
    return (model, tokenizer)


@pytest.fixture
def local_transformer_model(
    test_data_dir: pathlib.Path,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    model = transformers.AutoModel.from_pretrained(test_data_dir / "roberta-minuscule")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        test_data_dir / "roberta-minuscule", use_fast=True, add_prefix_space=True
    )
    return (model, tokenizer)


# TODO: do we really need lazy fixture here?
@pytest.fixture(
    params=[
        lazy_fixture("local_transformer_model"),
        lazy_fixture("remote_transformer_model"),
    ],
)
def transformer_model(
    request,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    return request.param


# NOTE: The transformer models are not reset between examples but that is *acceptable* and makes
# tests faster
@settings(deadline=2000, suppress_health_check=[HealthCheck.function_scoped_fixture])
# FIXME: should we really skip control characters and whitespaces? We do now because most 🤗
# tokenizers strip them out instead of rendering them as unk. Also formatters ? This forbids ZWNJ??
@given(
    data=st.data(),
    subwords_reduction=st.one_of([st.just("first"), st.just("mean")]),
    test_text=st.lists(
        st.text(
            alphabet=st.characters(
                blacklist_categories=["Cc", "Cf", "Cs", "Cn", "Co", "Zl", "Zp", "Zs"]
            ),
            min_size=1,
        ),
        min_size=1,
    ),
    weight_layers=st.booleans(),
)
def test_bert_embeddings_create_save_load(
    data: st.DataObject,
    subwords_reduction: Literal["first", "mean"],
    test_text: list[str],
    transformer_model: tuple[
        transformers.PreTrainedModel,
        transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    ],
    weight_layers: bool,
):
    model, tokenizer = transformer_model
    max_num_layers = int(
        min(
            getattr(model.config, param_name, math.inf)
            for param_name in ("num_layers", "n_layers", "num_hidden_layers")
        )
    )
    layers = data.draw(
        st.one_of(
            [
                st.none(),
                st.lists(
                    st.integers(min_value=-max_num_layers, max_value=max_num_layers - 1),
                    min_size=1,
                ),
            ]
        )
    )
    lexer = lexers.BertLexer(
        layers=layers,
        model=model,
        subwords_reduction=subwords_reduction,
        tokenizer=tokenizer,
        weight_layers=weight_layers,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        lexer.save(tmp_path, save_weights=True)
        # The only way I have found to enforce the identity of attention implementation in
        # roundtripping hf models. ugh.
        model_config_path = tmp_path / "model" / "config.json"
        model_config_path.write_text(
            json.dumps(
                {
                    **json.loads(model_config_path.read_text()),
                    "attn_implementation": lexer.model.config._attn_implementation,
                    "_attn_implementation_autoset": False,
                }
            )
        )
        reloaded = lexers.BertLexer.load(tmp_path)
        # Should always be true but lol
        assert reloaded.model.config._attn_implementation == lexer.model.config._attn_implementation
    # TODO: there must be a better way to deal with roots
    test_text = ["", *test_text]
    lexer.eval()
    reloaded.eval()
    try:
        lexer_encoding = lexer.encode(test_text)
    except lexers.LexingError:
        with pytest.raises(lexers.LexingError):
            reloaded_encoding = reloaded.encode(test_text)
        return
    # This is really cheesy, there must be a way to avoid repetition
    reloaded_encoding = reloaded.encode(test_text)
    # Should we do an assert on the encodings here? I don't think so since we check after batching
    # but.
    lexer_batch = lexer.make_batch([lexer_encoding])
    reloaded_batch = reloaded.make_batch([reloaded_encoding])
    assert lexer_batch.subword_alignments == reloaded_batch.subword_alignments
    assert lexer_batch.encoding.data.keys() == reloaded_batch.encoding.data.keys()
    for k, v in lexer_batch.encoding.data.items():
        assert torch.equal(v, reloaded_batch.encoding.data[k])
    with torch.inference_mode():
        orig_encoding = lexer(lexer_batch)
        reloaded_encoding = reloaded(reloaded_batch)
        assert torch.equal(orig_encoding, reloaded_encoding)
