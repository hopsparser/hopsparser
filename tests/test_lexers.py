import math
import pathlib
import tempfile
from typing import List, Literal, Tuple

import fasttext
import pytest
import torch
import transformers
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest_lazyfixture import lazy_fixture

from hopsparser import lexers


@given(
    char_embeddings_dim=st.integers(min_value=1, max_value=512),
    chars=st.lists(st.characters(), min_size=1),
    half_output_dim=st.integers(min_value=1, max_value=256),
    special_tokens=st.lists(st.text(min_size=2), max_size=8),
    test_text=st.lists(st.text(min_size=1), min_size=1),
)
def test_char_rnn_create_save_load(
    char_embeddings_dim: int,
    chars: List[str],
    half_output_dim: int,
    special_tokens: List[str],
    test_text: List[str],
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
    orig_encoding = char_lexer(char_lexer.make_batch([char_lexer.encode(test_text)]))
    reloaded.eval()
    reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
    assert torch.equal(orig_encoding, reloaded_encoding)


@settings(deadline=1000)
@given(
    special_tokens=st.lists(st.text(min_size=2), max_size=8),
    train_text=st.lists(
        st.lists(
            st.text(alphabet=st.characters(blacklist_categories=["Zs", "C"]), min_size=1),
            min_size=1,
        ),
        min_size=1,
    ),
    test_text=st.lists(st.text(min_size=1), min_size=1),
)
def test_fasttext_train_create_save_load(
    special_tokens: List[str],
    train_text: List[List[str]],
    test_text: List[str],
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        # We need to do this because of the hardcoded fasttext hp in from_raw
        train_txt_path = tmp_path / "train.txt"
        train_txt_path.write_text("\n".join(" ".join(w for w in s) for s in train_text))
        # This is a very very bad Fasttext model
        model = fasttext.train_unsupervised(
            str(train_txt_path),
            model="skipgram",
            minCount=0,
            epoch=1,
            ws=1,
            neg=1,
            bucket=128,
        )
        fasttext_lexer = lexers.FastTextLexer(
            model,
            special_tokens=special_tokens,
        )
        fasttext_lexer.save(tmp_path, save_weights=True)
        reloaded = lexers.FastTextLexer.load(tmp_path)
    assert fasttext_lexer.special_tokens == reloaded.special_tokens
    fasttext_lexer.eval()
    orig_encoding = fasttext_lexer(fasttext_lexer.make_batch([fasttext_lexer.encode(test_text)]))
    reloaded.eval()
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
    test_text: List[str],
    word_dropout: float,
    words: List[str],
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
    orig_encoding = lexer(lexer.make_batch([lexer.encode(test_text)]))
    reloaded.eval()
    reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
    assert torch.equal(orig_encoding, reloaded_encoding)


@pytest.fixture(params=["lgrobol/flaubert-minuscule"])
def remote_transformer_model(
    request,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    model = transformers.AutoModel.from_pretrained(request.param)
    tokenizer = transformers.AutoTokenizer.from_pretrained(request.param, use_fast=True)
    return (model, tokenizer)


@pytest.fixture
def local_transformer_model(
    test_data_dir: pathlib.Path,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    model = transformers.AutoModel.from_pretrained(test_data_dir / "roberta-minuscule")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        test_data_dir / "roberta-minuscule", use_fast=True, add_prefix_space=True
    )
    return (model, tokenizer)


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("local_transformer_model"),
        lazy_fixture("remote_transformer_model"),
    ],
)
def transformer_model(
    request,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    return request.param


@settings(deadline=2000)
# FIXME: should we really skip control characters and whitespaces? We do now because most 🤗
# tokenizers strip them out instead of rendering them as unk
@given(
    data=st.data(),
    subwords_reduction=st.one_of([st.just("first"), st.just("mean")]),
    test_text=st.lists(
        st.text(
            alphabet=st.characters(blacklist_categories=["Cc", "Cs", "Cn", "Zl", "Zp", "Zs"]),
            min_size=1,
        ),
        min_size=1,
    ),
    weight_layers=st.booleans(),
)
def test_bert_embeddings_create_save_load(
    data: st.DataObject,
    subwords_reduction: Literal["first", "mean"],
    test_text: List[str],
    transformer_model: Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase],
    weight_layers: bool,
):
    model, tokenizer = transformer_model
    max_num_layers = min(
        getattr(model.config, param_name, math.inf)
        for param_name in ("num_layers", "n_layers", "num_hidden_layers")
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
        reloaded = lexers.BertLexer.load(tmp_path)
    # TODO: there must be a better way to deal with roots
    test_text = ["", *test_text]
    lexer.eval()
    reloaded.eval()
    try:
        orig_encoding = lexer(lexer.make_batch([lexer.encode(test_text)]))
    except lexers.LexingError:
        with pytest.raises(lexers.LexingError):
            reloaded.encode(test_text)
    else:
        reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
        assert torch.equal(orig_encoding, reloaded_encoding)
