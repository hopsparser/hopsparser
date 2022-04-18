import pathlib
import tempfile
from typing import List, Tuple

import torch.cuda

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings
from torch.testing import assert_close

from hopsparser.parser import BiAffineParser
from hopsparser.deptree import DepGraph
from hopsparser.lexers import LexingError

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")


@pytest.mark.parametrize("source_device", devices)
@pytest.mark.parametrize("target_device", devices)
def test_initialize_save_load(
    source_device: str,
    target_device: str,
    train_config: pathlib.Path,
    treebank: pathlib.Path,
):
    source_device_d = torch.device(source_device)
    target_device_d = torch.device(target_device)
    parser = BiAffineParser.initialize(
        config_path=train_config,
        treebank=list(DepGraph.read_conll(open(treebank))),
    )
    parser.to(source_device_d)
    for _, p in parser.named_parameters():
        assert p.device == source_device_d
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        parser.save(tmp_path, save_weights=True)
        _ = BiAffineParser.load(tmp_path)
    parser.to(target_device_d)
    for _, p in parser.named_parameters():
        assert p.device == target_device_d


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


# FIXME: this should be generated in hypothesis to allow config variation
@pytest.fixture(scope="session")
def parser(
    train_config: pathlib.Path,
    treebank: pathlib.Path,
):
    parser = BiAffineParser.initialize(
        config_path=train_config,
        treebank=list(DepGraph.read_conll(open(treebank))),
    )
    return parser


@pytest.mark.parametrize("device", devices)
@settings(deadline=8192)
# FIXME: should we really skip control characters and whitespaces? We do now because most 🤗
# tokenizers strip them out instead of rendering them as unk (see also test_lexers)
@given(
    stable_text=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=["Zs", "C"]), min_size=1),
        min_size=1, max_size=32,
    ),
    distractor_text_1=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=["Zs", "C"]), min_size=1),
        min_size=1, max_size=32,
    ),
    distractor_text_2=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=["Zs", "C"]), min_size=1),
        min_size=1, max_size=32,
    ),
)
def test_batch_invariance(
    device: str,
    parser: BiAffineParser,
    stable_text: List[str],
    distractor_text_1: List[str],
    distractor_text_2: List[str],
):
    parser = parser.to(device)
    parser.eval()
    try:
        encoded_stable_text = parser.encode_sentence(stable_text, strict=True)
        encoded_distractor_text_1 = parser.encode_sentence(
            distractor_text_1, strict=True
        )
        encoded_distractor_text_2 = parser.encode_sentence(
            distractor_text_2, strict=True
        )
    except LexingError:
        assume(False)
    with torch.no_grad():
        stable_length = len(stable_text) + 1
        batch_stable = parser.batch_sentences([encoded_stable_text])
        text_s1 = [encoded_stable_text, encoded_distractor_text_1]
        text_1s = [encoded_distractor_text_1, encoded_stable_text]
        text_s2 = [encoded_stable_text, encoded_distractor_text_2]
        ref_tagger_scores: torch.Tensor
        ref_arc_scores: torch.Tensor
        ref_lab_scores: torch.Tensor
        ref_tagger_scores, ref_arc_scores, ref_lab_scores = parser(
            batch_stable.encodings, batch_stable.sent_lengths
        )
        for text, idx in ((text_s1, 0), (text_1s, 1), (text_s2, 0)):
            batch = parser.batch_sentences(text)
            tagger_scores: torch.Tensor
            arc_scores: torch.Tensor
            lab_scores: torch.Tensor
            tagger_scores, arc_scores, lab_scores = parser(
                batch.encodings, batch.sent_lengths
            )
            assert_close(
                ref_tagger_scores[0, :stable_length, :],
                tagger_scores[idx, :stable_length, :],
            )
            assert_close(
                ref_arc_scores[0, :stable_length, :stable_length],
                arc_scores[idx, :stable_length, :stable_length],
            )
            assert_close(
                ref_lab_scores[0, :stable_length, :stable_length, :],
                lab_scores[idx, :stable_length, :stable_length, :],
            )


@pytest.mark.parametrize("device", devices)
@settings(deadline=8192)
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
