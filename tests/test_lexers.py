import pathlib
import tempfile
from typing import List

import fasttext
import torch
from hypothesis import given, settings, strategies as st

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


@settings(deadline=500)
@given(
    special_tokens=st.lists(st.text(min_size=2), max_size=8),
    train_text=st.lists(st.lists(st.text(min_size=1), min_size=1), min_size=1),
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
            minCount=1,
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
    orig_encoding = fasttext_lexer(
        fasttext_lexer.make_batch([fasttext_lexer.encode(test_text)])
    )
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
        words_padding_idx=0,
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
