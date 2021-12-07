import pathlib
import tempfile
from typing import List, Optional, Tuple

import torch.cuda

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from hopsparser.parser import BiAffineParser
from hopsparser.deptree import DepGraph

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")


def test_initialize_save_load(
    train_config: pathlib.Path,
    treebank: pathlib.Path,
):
    parser = BiAffineParser.initialize(
        config_path=train_config,
        treebank=list(DepGraph.read_conll(open(treebank))),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        parser.save(tmp_path, save_weights=True)
        _ = BiAffineParser.load(tmp_path)


@pytest.fixture(scope="session")
def parser_and_reload(
    train_config: pathlib.Path,
    treebank: pathlib.Path,
):
    parser = BiAffineParser.initialize(
        config_path=train_config,
        treebank=list(DepGraph.read_conll(open(treebank))),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        parser.save(tmp_path, save_weights=True)
        reloaded = BiAffineParser.load(tmp_path)
    return parser, reloaded


@pytest.mark.parametrize("device", devices)
@settings(deadline=2000)
# FIXME: should we really skip control characters and whitespaces? We do now because most 🤗
# tokenizers strip them out instead of rendering them as unk (see also test_lexers)
@given(
    test_text=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=["Zs", "C"]), min_size=1),
        min_size=1,
    ),
)
def test_save_load_idempotency(
    device: str,
    parser_and_reload: Tuple[BiAffineParser, BiAffineParser],
    test_text: List[str],
):

    parser, reloaded = parser_and_reload
    parser = parser.to(device)
    reloaded = reloaded.to(device)
    test_sent_as_batch = [" ".join(test_text)]
    original_parse = parser.parse(test_sent_as_batch, raw=True, strict=False)
    reloaded_parse = reloaded.parse(test_sent_as_batch, raw=True, strict=False)
    original_parsed_conll = "\n\n".join(t.to_conllu() for t in original_parse)
    reloaded_parsed_conll = "\n\n".join(t.to_conllu() for t in reloaded_parse)
    assert reloaded_parsed_conll == original_parsed_conll
