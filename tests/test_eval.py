import pytest
from hopsparser.conll2018_eval import UDError
from hopsparser.conll2018_eval import UDRepresentation, evaluate, load_conllu

from hypothesis import given
from hypothesis import strategies as st


@st.composite
def trees(draw: st.DrawFn, words: st.SearchStrategy[list[str]]) -> UDRepresentation:
    """Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors."""
    lines, num_words = [], 0
    for w in draw(words):
        parts = w.split()
        if len(parts) == 1:
            num_words += 1
            lines.append(f"{num_words}\t{parts[0]}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
        else:
            lines.append(
                f"{num_words + 1}-{num_words + len(parts)}\t{parts[0]}\t_\t_\t_\t_\t_\t_\t_\t_"
            )
            for part in parts[1:]:
                num_words += 1
                lines.append(f"{num_words}\t{part}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
    return load_conllu((*lines, "\n"))


def validate_correct(gold: UDRepresentation, system: UDRepresentation, correct: int):
    metrics = evaluate(gold, system)
    assert (metrics["Words"].precision, metrics["Words"].recall, metrics["Words"].f1) == (
        correct / len(system.words),
        correct / len(gold.words),
        2 * correct / (len(gold.words) + len(system.words)),
    )


@given(
    gold=trees(words=st.just(["a"])),
    system=trees(words=st.just(["b"])),
)
def test_exception(gold: UDRepresentation, system: UDRepresentation):
    with pytest.raises(UDError):
        evaluate(gold, system)


@given(
    representation=trees(
        words=st.one_of(
            st.just(["a"]),
            st.just(["a", "b", "c"]),
            st.lists(
                st.text(
                    alphabet=st.characters(blacklist_categories=["Zl", "Zp"]),
                    min_size=1,
                ).filter(lambda s: not s.isspace()),
                min_size=1,
            ),
        ),
    )
)
def test_equal(representation: UDRepresentation):
    validate_correct(representation, representation, len(representation.words))


@given(
    args=st.one_of([
        st.tuples(
            trees(words=st.just(["abc a b c"])),
            trees(words=st.just(["a", "b", "c"])),
            st.just(3),
        ),
        st.tuples(
            trees(words=st.just(["a", "bc b c", "d"])),
            trees(words=st.just(["a", "b", "c", "d"])),
            st.just(4),
        ),
        st.tuples(
            trees(words=st.just(["abcd a b c d"])),
            trees(words=st.just(["ab a b", "cd c d"])),
            st.just(4),
        ),
        st.tuples(
            trees(words=st.just(["abc a b c", "de d e"])),
            trees(words=st.just(["a", "bcd b c d", "e"])),
            st.just(5),
        ),
    ]),
)
def test_multiwords(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    validate_correct(gold, system, correct)


@given(
    args=st.one_of([
        st.tuples(
            trees(words=st.just(["abcd"])),
            trees(words=st.just(["a", "b", "c", "d"])),
            st.just(0),
        ),
        st.tuples(
            trees(words=st.just(["abc", "d"])),
            trees(words=st.just(["a", "b", "c", "d"])),
            st.just(1),
        ),
        st.tuples(
            trees(words=st.just(["a", "bc", "d"])),
            trees(words=st.just(["a", "b", "c", "d"])),
            st.just(2),
        ),
        st.tuples(
            trees(words=st.just(["a", "bc b c", "d"])),
            trees(words=st.just(["a", "b", "cd"])),
            st.just(2),
        ),
        st.tuples(
            trees(words=st.just(["abc a BX c", "def d EX f"])),
            trees(words=st.just(["ab a b", "cd c d", "ef e f"])),
            st.just(4),
        ),
        st.tuples(
            trees(words=st.just(["ab a b", "cd bc d"])),
            trees(words=st.just(["a", "bc", "d"])),
            st.just(2),
        ),
        st.tuples(
            trees(words=st.just(["a", "bc b c", "d"])),
            trees(words=st.just(["ab AX BX", "cd CX a"])),
            st.just(1),
        ),
    ]),
)
def test_alignment(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    validate_correct(gold, system, correct)
