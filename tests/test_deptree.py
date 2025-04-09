import pathlib

from common import sent_conllus
from hypothesis import strategies as st
from hypothesis import given

from hopsparser import deptree


def test_conllu_idempotency_const(treebank_path: pathlib.Path):
    treebank_content = treebank_path.read_text()
    treebank_obj = list(deptree.DepGraph.read_conll(treebank_content.splitlines()))
    roundtripped_treebank_content = "\n\n".join(t.to_conllu() for t in treebank_obj)
    # Strip because UD files are supposed to be terminated by a blank line but we don't really want
    # to check that
    assert treebank_content.strip() == roundtripped_treebank_content.strip()


@given(
    treebank=st.lists(sent_conllus(), min_size=1).map(
        lambda treebank: [line for tree in treebank for line in (*tree, "")]
    )
)
def test_conllu_idempotency(treebank: list[str]):
    # We don't just test this for a single tree because we want to ensure that tree separation works
    treebank_obj = list(deptree.DepGraph.read_conll(treebank))
    roundtripped_treebank = [
        line for t in treebank_obj for line in (*t.to_conllu().splitlines(), "")
    ]
    assert treebank == roundtripped_treebank
