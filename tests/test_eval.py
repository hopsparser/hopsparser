import heapq
from typing import Iterable, Sequence, cast
import numpy as np
import pytest
from hopsparser.conll2018_eval import UDError
from hopsparser.conll2018_eval import UDRepresentation, evaluate, load_conllu

from hypothesis import assume, given
from hypothesis import strategies as st


def seq_to_heads(
    seq: np.ndarray[tuple[int], np.dtype[np.intp]], root: np.intp
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Given a Prüfer sequence for a rooted tree, return the corresponding arborescence as an
    array `heads` encoding all the `(i, heads[i])` directed arcs. `heads[root]` is set to -1.

    The Prüfer sequence for a rooted tree is build exactly like a regular Prüfer sequence, except
    that at each step, instead of removing the leaf with the lowest index, you remove the leaf with
    the lowest index *that is not the root*.
    """
    n = seq.shape[0] + 2

    degrees = np.ones(n, dtype=np.intp)
    values, counts = np.unique_counts(seq)
    degrees[values] += counts
    # This is not actually the degree of the root, but this way it can't never enter the leaves heap
    degrees[root] = 0
    # We need a list for heapification. You'd THINK we could store degrees - 1 to make this terser
    # but actually numpy doesn't have a zero_loc function lol. The type is a union because at first
    # we'll get Python int but later we won't be bothering casting the np int we'll be getting.
    leaves = cast(list[int | np.intp], np.flatnonzero(degrees == 1).tolist())
    # that's a min-heap, ty Python
    heapq.heapify(leaves)

    heads = np.full(n, fill_value=-1, dtype=np.intp)

    # We are replaying the sequence building: at each step, the node `i` that was popped and whose
    # sole neighbour `j` got put in the sequence was the leaf with the lowest index that is not the
    # root. So if we see `j` in the sequence, we only need to find `i` in the same way and set
    # `heads[i] = j`. Since `i` is then removed from the leaves heap, we can't overwrite it.
    # At the end, we will have set `n-2` arcs
    for h in seq:
        leaf = heapq.heappop(leaves)
        heads[leaf] = h
        # We have seen all the neighbours of `head` but one, and it is not the root (otherwise its
        # "degree" would be 0, so it becomes a leaf.
        if degrees[h] == 2:
            # Slightly less efficient than heapreplace but more legible
            heapq.heappush(leaves, h)
        elif h != root:
            degrees[h] -= 1

    # There's two nodes left unpopped after building the Prüfer sequence: the root, and that last
    # leaf that's still in the heap and whose head can only be the root.
    heads[leaves[0]] = root

    # At this point we have set `n-1` arcs, so we have retrieved all the arcs of the original tree
    # and we only need to be sure that they are in the right direction for our arborescence, which
    # is the case: at this point the neighbours of a node `i` are exactly the nodes `j` such that
    # `heads[j] == i`, plus `heads[i]` if `i != root`. Now if we say that a non-root node `i` is
    # *correctly oriented* iff `heads[i]` is its actual head in the arborescence, then:
    #
    # - Every child `i` of `root` is correctly oriented: since it's a neighbour of `root`, we must
    #   have `heads[i] == root`.
    # - If a non-root node `i` is correctly oriented, then for every child `j` of `i`, since `j` is
    #   neighbour of `i` and `heads[i] != j` (because `i` is correctly oriented), we must have
    #   `heads[j] == i`. Therefore `j` is correctly oriented.
    #
    # Therefore, by induction, all the non-root nodes are correctly oriented, so we have the correct
    # head sequence.

    return heads


def heads_to_seq(
    heads: np.ndarray[tuple[int], np.dtype[np.intp]], root: np.intp
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Given an arborescence as an array `heads` encoding all the `(i, heads[i])` directed arcs,
    return the corresponding Prüfer sequence. `heads[root]` must be `-1`.

    The Prüfer sequence for a rooted tree is build exactly like a regular Prüfer sequence, except
    that at each step, instead of removing the leaf with the lowest index, you remove the leaf with
    the lowest index *that is not the root*.
    """
    n = heads.shape[0]
    degrees = np.ones(n, dtype=np.intp)
    values, counts = np.unique_counts(heads)
    # Since the root is -1 and the values are sorted, we can safely escape this way
    degrees[values[1:]] += counts[1:]
    degrees[root] = 0
    leaves = cast(list[int], np.flatnonzero(degrees == 1).tolist())
    heapq.heapify(leaves)

    seq = np.empty(n - 2, dtype=np.intp)

    for i in range(n - 2):
        leaf = heapq.heappop(leaves)
        h = heads[leaf]
        seq[i] = h
        # We have seen all the neighbours of `head` but one, and it is not the root (otherwise its
        # "degree" would be 0, so it becomes a leaf.
        if degrees[h] == 2:
            # Slightly less efficient than heapreplace but more legible
            heapq.heappush(leaves, h)
        elif h != root:
            degrees[h] -= 1

    return seq


tokens_strat = st.text(
    alphabet=st.characters(blacklist_characters=["\n", "\r", "\t"]),
    min_size=1,
).filter(lambda s: not s.isspace())


sentences_strat = st.lists(
    st.one_of([tokens_strat, st.tuples(tokens_strat, st.lists(tokens_strat))]),
    min_size=1,
)


@st.composite
def conllus(
    draw: st.DrawFn, tokens: st.SearchStrategy[Iterable[str | tuple[str, Sequence[str]]]]
) -> list[str]:
    """Generate fake conllu files"""
    lines, num_words = [], 0
    for tok in draw(tokens):
        if isinstance(tok, str):
            num_words += 1
            lines.append(f"{num_words}\t{tok}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")
        else:
            surface_token, parts = tok
            assume(len(parts) >= 2)
            lines.append(
                f"{num_words + 1}-{num_words + len(parts)}\t{surface_token}\t_\t_\t_\t_\t_\t_\t_\t_"
            )
            for p in parts:
                num_words += 1
                lines.append(f"{num_words}\t{p}\t_\t_\t_\t_\t{int(num_words > 1)}\t_\t_\t_")

    return [*lines, "\n"]


@st.composite
def trees(
    draw: st.DrawFn, tokens: st.SearchStrategy[Iterable[str | tuple[str, Sequence[str]]]]
) -> UDRepresentation:
    lines = draw(conllus(tokens=tokens))
    return load_conllu(lines)


@given(
    lines=conllus(
        tokens=st.one_of(
            st.just(["a"]),
            st.just(["a", "b", "c"]),
            sentences_strat,
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
                    alphabet=st.characters(blacklist_characters=["\n", "\r", "\t"]),
                    min_size=1,
                ).filter(lambda s: not s.isspace()),
                min_size=1,
            ),
        ),
    )
)
def test_equal(representation: UDRepresentation):
    metrics = evaluate(representation, representation)
    assert metrics["Words"].correct == len(representation.words)


@given(
    args=st.one_of([
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
def test_multiwords(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    metrics = evaluate(gold, system)
    assert metrics["Words"].correct == correct


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
    ]),
)
def test_alignment(args: tuple[UDRepresentation, UDRepresentation, int]):
    gold, system, correct = args
    metrics = evaluate(gold, system)
    assert metrics["Words"].correct == correct
