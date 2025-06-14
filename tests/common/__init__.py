import heapq
from typing import Sequence, cast

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st


def seq_to_heads(
    seq: Sequence[int], root: np.intp | int
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Given a Prüfer sequence for a rooted tree, return the corresponding arborescence as an
    array `heads` encoding all the `(i, heads[i])` directed arcs. `heads[root]` is set to -1. Note
    that *any* sequence of length `n-2` of integers between `0` and `n-1` is a valid Prüfer
    sequence.

    The Prüfer sequence for a rooted tree is built exactly like a regular Prüfer sequence, except
    that at each step, instead of removing the leaf with the lowest index, you remove the leaf with
    the lowest index *that is not the root*.
    """
    _seq = np.asarray(seq, dtype=np.intp)
    n = _seq.shape[0] + 2

    degrees = np.ones(n, dtype=np.intp)
    values, counts = np.unique_counts(_seq)
    degrees[values] += counts
    # This is not actually the degree of the root, but this way it can't ever enter the leaves heap,
    # since we make that happen when `degrees[i]` is about to become 1.
    degrees[root] = 0
    # We need a list for heapification. You'd THINK we could store degrees - 1 to make this terser
    # but actually numpy doesn't have a zero_loc function lol. The type is a union because at first
    # we'll get Python ints but later we won't be bothering casting the np int we'll be getting.
    leaves = cast(list[int | np.intp], np.flatnonzero(degrees == 1).tolist())
    # that's a min-heap, ty Python
    heapq.heapify(leaves)

    heads = np.full(n, fill_value=-1, dtype=np.intp)

    # We are replaying the sequence building: at each step, the node `i` that was popped and whose
    # sole neighbour `j` got put in the sequence was the leaf with the lowest index that is not the
    # root. So if we see `j` in the sequence, we only need to find `i` in the same way and set
    # `heads[i] = j`. Since `i` is then removed from the leaves heap and can't ever re-enter it
    # (because it can't be present in the rest of seq), we can't overwrite it. At the end, we will
    # have set `n-2` arcs
    for h in _seq:
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
    heads: Sequence[int], root: int | np.intp
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Given an arborescence as an array `heads` encoding all the `(i, heads[i])` directed arcs,
    return the corresponding Prüfer sequence. `heads[root]` must be `-1`.

    The Prüfer sequence for a rooted tree is built exactly like a regular Prüfer sequence, except
    that at each step, instead of removing the leaf with the lowest index, you remove the leaf with
    the lowest index *that is not the root*.
    """
    _heads = np.asarray(heads, dtype=np.intp)
    n = _heads.shape[0]
    degrees = np.ones(n, dtype=np.intp)
    values, counts = np.unique_counts(_heads)
    # Since the root is -1 and the values are sorted, we can safely escape this way
    degrees[values[1:]] += counts[1:]
    degrees[root] = 0
    leaves = cast(list[int], np.flatnonzero(degrees == 1).tolist())
    heapq.heapify(leaves)

    seq = np.empty(n - 2, dtype=np.intp)

    for i in range(n - 2):
        leaf = heapq.heappop(leaves)
        h = _heads[leaf]
        seq[i] = h
        # We have seen all the neighbours of `head` but one, and it is not the root (otherwise its
        # "degree" would be 0), so it becomes a leaf.
        if degrees[h] == 2:
            # Slightly less efficient than heapreplace but more legible
            heapq.heappush(leaves, h)
        elif h != root:
            degrees[h] -= 1

    return seq


# We only forbid all newlines (see <https://docs.python.org/3/library/stdtypes.html#str.splitlines>
# and the tabulator
conllu_filled_column_st = st.text(
    alphabet=st.characters(
        blacklist_characters=[
            "\t",
            "\n",
            "\r",
            "\N{LINE TABULATION}",
            "\N{FORM FEED}",
            "\N{FILE SEPARATOR}",
            "\N{GROUP SEPARATOR}",
            "\N{RECORD SEPARATOR}",
            "\N{NEXT LINE}",
            "\N{LINE SEPARATOR}",
            "\N{PARAGRAPH SEPARATOR}",
        ]
    ),
    min_size=1,
).filter(lambda s: not s.isspace())

conllu_column_st = st.one_of([st.just("_"), conllu_filled_column_st])

conllu_token_lists = st.lists(
    st.one_of([
        conllu_filled_column_st,
        st.tuples(conllu_filled_column_st, st.lists(conllu_filled_column_st, min_size=2)),
    ]),
    min_size=1,
)


@st.composite
def conllu_lines(draw: st.DrawFn, indice: str, form: str, head: str | None) -> str:
    if head is None:
        return "\t".join([
            indice,
            form,
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
            draw(conllu_column_st),
        ])
    return "\t".join([
        indice,
        form,
        draw(conllu_column_st),
        draw(conllu_column_st),
        draw(conllu_column_st),
        draw(conllu_column_st),
        head,
        draw(conllu_column_st),
        draw(conllu_column_st),
        draw(conllu_column_st),
    ])


@st.composite
def sent_conllus(
    draw: st.DrawFn,
    tokens: st.SearchStrategy[Sequence[str | tuple[str, Sequence[str]]]] | None = None,
) -> list[str]:
    """Generate fake conllu files"""
    if tokens is None:
        tokens_lst = draw(conllu_token_lists)
    else:
        tokens_lst = draw(tokens)
    n_words = sum(len(t[1]) if isinstance(t, tuple) else 1 for t in tokens_lst)
    if n_words == 1:
        heads = iter([0])
    else:
        heads = iter(
            seq_to_heads(
                draw(
                    st.lists(
                        st.integers(0, n_words - 1), min_size=n_words - 2, max_size=n_words - 2
                    )
                ),
                root=draw(st.integers(0, n_words - 1)),
            )
            + 1
        )
    word_idx = 0
    # A sentence can have metadata and/or comments
    lines = draw(
        st.lists(
            st.text(alphabet=st.characters(blacklist_categories=["Cc", "Zl"])).map(
                lambda s: f"#{s}"
            )
        )
    )
    for tok in tokens_lst:
        if isinstance(tok, str):
            word_idx += 1
            lines.append(draw(conllu_lines(indice=str(word_idx), form=tok, head=str(next(heads)))))
        else:
            surface_token, parts = tok
            assume(len(parts) >= 2)
            lines.append(
                draw(
                    conllu_lines(
                        indice=f"{word_idx + 1}-{word_idx + len(parts)}",
                        form=surface_token,
                        head=None,
                    )
                )
            )
            for p in parts:
                word_idx += 1
                lines.append(
                    draw(conllu_lines(indice=str(word_idx), form=p, head=str(next(heads))))
                )

    return lines
