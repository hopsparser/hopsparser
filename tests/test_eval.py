import io

import pytest
from hopsparser.conll2018_eval import UDError
from hopsparser.conll2018_eval import UDRepresentation, evaluate, load_conllu

from hypothesis import given
from hypothesis import strategies as st


def load_words(words: list[str]) -> UDRepresentation:
    """Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors."""
    lines, num_words = [], 0
    for w in words:
        parts = w.split()
        if len(parts) == 1:
            num_words += 1
            lines.append(f"{num_words}\t{parts[0]}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
        else:
            lines.append(
                f"{num_words + 1}-{num_words + len(parts) - 1}\t{parts[0]}\t_\t_\t_\t_\t_\t_\t_\t_"
            )
            for part in parts[1:]:
                num_words += 1
                lines.append(f"{num_words}\t{part}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
    return load_conllu(io.StringIO("\n".join(lines + ["\n"])))


def validate_correct(gold: UDRepresentation, system: UDRepresentation, correct: int):
    metrics = evaluate(gold, system)
    assert (metrics["Words"].precision, metrics["Words"].recall, metrics["Words"].f1) == (
        correct / len(system.words),
        correct / len(gold.words),
        2 * correct / (len(gold.words) + len(system.words)),
    )


@given(
    gold=st.builds(load_words, st.just(["a"])),
    system=st.builds(load_words, st.just(["b"])),
)
def test_exception(gold: UDRepresentation, system: UDRepresentation):
    with pytest.raises(UDError):
        evaluate(gold, system)


@given(representation=st.builds(load_words, st.one_of(st.just(["a"]), st.just(["a", "b", "c"]))))
def test_equal(representation: UDRepresentation):
    validate_correct(representation, representation, len(representation.words))


@given(
    args=st.builds(
        (lambda t: (load_words(t[0]), load_words(t[1]), t[2])),
        st.one_of([
            st.just((["abc a b c"], ["a", "b", "c"], 3)),
            st.just((["a", "bc b c", "d"], ["a", "b", "c", "d"], 4)),
            st.just((["abcd a b c d"], ["ab a b", "cd c d"], 4)),
            st.just((["abc a b c", "de d e"], ["a", "bcd b c d", "e"], 5)),
        ]),
    )
)
def test_multiwords(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    validate_correct(gold, system, correct)


@given(
    args=st.builds(
        (lambda t: (load_words(t[0]), load_words(t[1]), t[2])),
        st.one_of([
            st.just((["abcd"], ["a", "b", "c", "d"], 0)),
            st.just((["abc", "d"], ["a", "b", "c", "d"], 1)),
            st.just((["a", "bc", "d"], ["a", "b", "c", "d"], 2)),
            st.just((["a", "bc b c", "d"], ["a", "b", "cd"], 2)),
            st.just((["abc a BX c", "def d EX f"], ["ab a b", "cd c d", "ef e f"], 4)),
            st.just((["ab a b", "cd bc d"], ["a", "bc", "d"], 2)),
            st.just((["a", "bc b c", "d"], ["ab AX BX", "cd CX a"], 1)),
        ]),
    )
)
def test_alignment(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    validate_correct(gold, system, correct)
