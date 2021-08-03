import pathlib
import tempfile
from typing import List

import torch
from hypothesis import given, strategies as st

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
    assert char_lexer.vocab == reloaded.vocab
    orig_encoding = char_lexer(char_lexer.make_batch([char_lexer.encode(test_text)]))
    reloaded_encoding = reloaded(reloaded.make_batch([reloaded.encode(test_text)]))
    assert torch.equal(orig_encoding, reloaded_encoding)
