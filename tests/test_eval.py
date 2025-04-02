import heapq
from typing import Iterable, Sequence
import numpy as np
import pytest
from hopsparser.conll2018_eval import UDError
from hopsparser.conll2018_eval import UDRepresentation, evaluate, load_conllu

from hypothesis import assume, given
from hypothesis import strategies as st


# This might be made more efficient by figuring out an alternative Prüfer scheme that directly gets
# the arcs with the right orientation????
def seq_to_heads(
    seq: np.ndarray[tuple[int], np.dtype[np.intp]], root: np.intp
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Given a Prüfer sequence for a tree and a root, return the corresponding arborescence as an
    array `heads` encoding all the `(i, heads[i])` directed arcs. `heads[root]` is set to -1."""
    n = seq.shape[0] + 2
    degrees = np.ones(n, dtype=np.intp)
    values, counts = np.unique_counts(seq)
    degrees[values] += counts
    # Need a list for heapification. You'd THINK we could store degrees - 1 to make this terser but
    # actually numpy doesn't have a zero_loc function lol.
    leaves = np.flatnonzero(degrees == 1).tolist()
    # that's a min-heap, ty Python
    heapq.heapify(leaves)

    # First determine the arcs for the case where the root is n-1, orienting them as `(i, arc[i])`
    arcs = np.full(n - 1, fill_value=n - 1, dtype=np.intp)

    for head in seq:
        leaf = heapq.heappop(leaves)
        arcs[leaf] = head
        if degrees[head] == 2:
            # Slightly less efficient than heapreplace but more legible
            heapq.heappush(leaves, head)
        else:
            degrees[head] -= 1

    # The arcs here give us the correct heads for a root at `n-1`: at this point the neighbours of a
    # non-root node `i` are exactly `arcs[i]` and the nodes `j` such that `arcs[j] == i`. Now if we
    # say that a non-root node `i` is *correctly oriented* iff `arcs[i]` is its head in the
    # `n-1`-rooted arborescence, then:
    # - Every child `i` of `n-1` is correctly oriented: since it's a neighbour of `n-1` and
    #   `arcs[n-1]` doesn't exist, we must have `arcs[i] == n-1`.
    # - If a node `i` is correctly oriented, then for every child `j` of `i`, since `j` is neighbour
    #   of `i` and `arcs[i] != j` (because `j` is not the head of `i`), we must have `arcs[j] == i`,
    #   and therefore `j` is correctly oriented.
    # And therefore, by induction, all the nodes are correctly oriented.

    if root == n - 1:
        return np.append(arcs, [-1])

    # Re-orient the arcs with a breadth-first search

    heads = np.full(n, fill_value=-1, dtype=np.intp)

    # special-casing the root case is slightly more efficient
    children = np.zeros_like(heads, dtype=np.bool)
    children[np.flatnonzero(arcs == root)] = True
    children[arcs[root]] = True
    heads[children] = root
    opened = np.flatnonzero(children).tolist()
    heapq.heapify(opened)

    while opened:
        current = heapq.heappop(opened)
        children.fill(False)
        children[np.flatnonzero(arcs == current)] = True
        if current != n - 1:
            children[arcs[current]] = True
        children[heads[current]] = False
        children_idx = np.flatnonzero(children)
        heads[children_idx] = current
        # we have no better heapextend method
        opened.extend(children_idx)
        heapq.heapify(opened)

    return heads


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
