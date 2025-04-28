from typing import Sequence
import pytest
from hopsparser.evaluator import UDError
from hopsparser.evaluator import UDRepresentation, evaluate, load_conllu

from common import sent_conllus, conllu_token_lists
from hypothesis import note, strategies as st
from hypothesis import given


@st.composite
def trees(
    draw: st.DrawFn, tokens: st.SearchStrategy[Sequence[str | tuple[str, Sequence[str]]]]
) -> UDRepresentation:
    lines = draw(sent_conllus(tokens=tokens))
    return load_conllu(lines)


@st.composite
def two_trees(
    draw: st.DrawFn, tokens: st.SearchStrategy[Sequence[str | tuple[str, Sequence[str]]]]
) -> tuple[UDRepresentation, UDRepresentation]:
    """Two trees over the same tokens"""
    actual_tokens = draw(tokens)
    lines_one = draw(sent_conllus(tokens=st.just(actual_tokens)))
    lines_two = draw(sent_conllus(tokens=st.just(actual_tokens)))
    return load_conllu(lines_one), load_conllu(lines_two)


@given(
    lines=sent_conllus(
        tokens=st.one_of(
            st.just(["a"]),
            st.just(["a", "b", "c"]),
            conllu_token_lists,
        ),
    )
)
def test_load_conllu(lines: list[str]):
    load_conllu(lines)


@given(
    gold=trees(tokens=st.just(["a"])),
    system=trees(tokens=st.just(["b"])),
)
def test_exception(gold: UDRepresentation, system: UDRepresentation):
    with pytest.raises(UDError):
        evaluate(gold, system)


@given(
    representation=trees(
        tokens=st.one_of(
            st.just(["a"]),
            st.just(["a", "b", "c"]),
            conllu_token_lists,
        ),
    )
)
def test_equal(representation: UDRepresentation):
    metrics = evaluate(representation, representation)
    assert metrics["Words"].correct == len(representation.words)
    for m, score in metrics.items():
        note(m)
        note(score)
        assert score.correct == score.gold_total == score.system_total
        if score.aligned_total is not None:
            assert score.correct == score.aligned_total
        if score.correct == 0:
            assert 0.0 == score.f1
        else:
            assert 1.0 == score.f1


@given(trees=two_trees(conllu_token_lists))
def test_boundaries(trees: tuple[UDRepresentation, UDRepresentation]):
    gold, system = trees
    metrics = evaluate(gold, system)
    for score in metrics.values():
        note(score)
        assert 0 <= score.correct <= score.gold_total
        assert 0 <= score.correct <= score.system_total
        assert 0.0 <= score.f1 <= 1.0


@given(
    args=st.one_of([
        st.tuples(
            trees(tokens=st.just(["abcd"])),
            trees(tokens=st.just(["a", "b", "c", "d"])),
            st.just(0),
        ),
        st.tuples(
            trees(tokens=st.just(["abc", "d"])),
            trees(tokens=st.just(["a", "b", "c", "d"])),
            st.just(1),
        ),
        st.tuples(
            trees(tokens=st.just(["a", "bc", "d"])),
            trees(tokens=st.just(["a", "b", "c", "d"])),
            st.just(2),
        ),
        st.tuples(
            trees(tokens=st.just(["a", ("bc", ["b", "c"]), "d"])),
            trees(tokens=st.just(["a", "b", "cd"])),
            st.just(2),
        ),
        st.tuples(
            trees(tokens=st.just([("abc", ["a", "BX", "c"]), ("def", ["d", "EX", "f"])])),
            trees(tokens=st.just([("ab", ["a", "b"]), ("cd", ["c", "d"]), ("ef", ["e", "f"])])),
            st.just(4),
        ),
        st.tuples(
            trees(tokens=st.just([("ab", ["a", "b"]), ("cd", ["bc", "d"])])),
            trees(tokens=st.just(["a", "bc", "d"])),
            st.just(2),
        ),
        st.tuples(
            trees(tokens=st.just(["a", ("bc", ["b", "c"]), "d"])),
            trees(tokens=st.just([("ab", ["AX", "BX"]), ("cd", ["CX", "a"])])),
            st.just(1),
        ),
        st.tuples(
            trees(tokens=st.just([("abc", ["a", "b"]), ("a", ["c", "a"])])),
            trees(tokens=st.just(["a", "b", "c", "a"])),
            st.just(3),
        ),
        # This next one is absurd but would fit the original algorithm
        # st.tuples(
        #     trees(tokens=st.just(["abcd", ("a", ["b", "c", "d"])])),
        #     trees(tokens=st.just(["a", "b", "c", "d", ("a", ["e", "f"])])),
        #     st.just(3),
        # ),
        # This one makes sense
        st.tuples(
            trees(tokens=st.just(["abcd", ("a", ["b", "c", "d"])])),
            trees(tokens=st.just(["a", "b", "c", "d", ("a", ["e", "f"])])),
            st.just(0),
        ),
        st.tuples(
            trees(tokens=st.just([("abc", ["a", "b", "c"])])),
            trees(tokens=st.just(["a", "b", "c"])),
            st.just(3),
        ),
        st.tuples(
            trees(tokens=st.just(["a", ("bc", ["b", "c"]), "d"])),
            trees(tokens=st.just(["a", "b", "c", "d"])),
            st.just(4),
        ),
        st.tuples(
            trees(tokens=st.just([("abcd", ["a", "b", "c", "d"])])),
            trees(tokens=st.just([("ab", ["a", "b"]), ("cd", ["c", "d"])])),
            st.just(4),
        ),
        st.tuples(
            trees(tokens=st.just([("abc", ["a", "b", "c"]), ("de", ["d", "e"])])),
            trees(tokens=st.just(["a", ("bcd", ["b", "c", "d"]), "e"])),
            st.just(5),
        ),
    ]),
)
def test_alignment(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    metrics = evaluate(gold, system)
    assert metrics["Words"].correct == correct
