import pytest
from hopsparser.conll2018_eval import UDError
from hopsparser.conll2018_eval import UDRepresentation, evaluate, load_conllu

from hypothesis import given
from hypothesis import strategies as st


# FIXME: this way of encoding generated multiword as `"surf part part…"` tokens is very annoying
@st.composite
def conllus(draw: st.DrawFn, tokens: st.SearchStrategy[list[str]]) -> list[str]:
    """Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors."""
    lines, num_words = [], 0
    for tok in draw(tokens):
        surface_token, *parts = tok.split()
        if parts:
            lines.append(
                f"{num_words + 1}-{num_words + len(parts)}\t{surface_token}\t_\t_\t_\t_\t_\t_\t_\t_"
            )
            for p in parts:
                num_words += 1
                lines.append(f"{num_words}\t{p}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
        else:
            num_words += 1
            lines.append(f"{num_words}\t{surface_token}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
    return [*lines, "\n"]


@st.composite
def trees(draw: st.DrawFn, tokens: st.SearchStrategy[list[str]]) -> UDRepresentation:
    lines = draw(conllus(tokens=tokens))
    return load_conllu(lines)


def validate_correct(gold: UDRepresentation, system: UDRepresentation, correct: int):
    metrics = evaluate(gold, system)
    assert (metrics["Words"].precision, metrics["Words"].recall, metrics["Words"].f1) == (
        correct / len(system.words),
        correct / len(gold.words),
        2 * correct / (len(gold.words) + len(system.words)),
    )


# TODO: add mwt testing
@given(
    lines=conllus(
        tokens=st.one_of(
            st.just(["a"]),
            st.just(["a", "b", "c"]),
            st.lists(
                st.text(
                    alphabet=st.characters(
                        blacklist_categories=["Z"], blacklist_characters=["\n", "\r", "\t"]
                    ),
                    min_size=1,
                ).filter(lambda s: not s.isspace()),
                min_size=1,
            ),
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


# TODO: add mwt testing
@given(
    representation=trees(
        tokens=st.one_of(
            st.just(["a"]),
            st.just(["a", "b", "c"]),
            st.lists(
                st.text(
                    alphabet=st.characters(
                        blacklist_categories=["Z"], blacklist_characters=["\n", "\r", "\t"]
                    ),
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
            trees(tokens=st.just(["abc a b c"])),
            trees(tokens=st.just(["a", "b", "c"])),
            st.just(3),
        ),
        st.tuples(
            trees(tokens=st.just(["a", "bc b c", "d"])),
            trees(tokens=st.just(["a", "b", "c", "d"])),
            st.just(4),
        ),
        st.tuples(
            trees(tokens=st.just(["abcd a b c d"])),
            trees(tokens=st.just(["ab a b", "cd c d"])),
            st.just(4),
        ),
        st.tuples(
            trees(tokens=st.just(["abc a b c", "de d e"])),
            trees(tokens=st.just(["a", "bcd b c d", "e"])),
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
            trees(tokens=st.just(["a", "bc b c", "d"])),
            trees(tokens=st.just(["a", "b", "cd"])),
            st.just(2),
        ),
        st.tuples(
            trees(tokens=st.just(["abc a BX c", "def d EX f"])),
            trees(tokens=st.just(["ab a b", "cd c d", "ef e f"])),
            st.just(4),
        ),
        st.tuples(
            trees(tokens=st.just(["ab a b", "cd bc d"])),
            trees(tokens=st.just(["a", "bc", "d"])),
            st.just(2),
        ),
        st.tuples(
            trees(tokens=st.just(["a", "bc b c", "d"])),
            trees(tokens=st.just(["ab AX BX", "cd CX a"])),
            st.just(1),
        ),
        st.tuples(
            trees(tokens=st.just(["abc a b", "a c a"])),
            trees(tokens=st.just(["a", "b", "c", "a"])),
            st.just(3),
        ),
        # This next one is absurd
        # st.tuples(
        #     trees(tokens=st.just(["abcd", "a b c d"])),
        #     trees(tokens=st.just(["a", "b", "c", "d", "a e f"])),
        #     st.just(3),
        # ),
        # This one makes sense
        st.tuples(
            trees(tokens=st.just(["abcd", "a b c d"])),
            trees(tokens=st.just(["a", "b", "c", "d", "a e f"])),
            st.just(0),
        ),
    ]),
)
def test_alignment(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    validate_correct(gold, system, correct)
