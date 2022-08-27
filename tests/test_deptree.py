import pathlib

from hopsparser import deptree


def test_conllu_idempotency(
    treebank: pathlib.Path,
):

    orig_treebank_content = treebank.read_text()
    treebank_obj = list(deptree.DepGraph.read_conll(orig_treebank_content.splitlines()))
    roundtripped_treebank_content = "\n\n".join(t.to_conllu() for t in treebank_obj)
    # Strip because UD files are supposed to be terminated by a blank line but we don't really want
    # to check that
    assert orig_treebank_content.strip() == roundtripped_treebank_content.strip()
