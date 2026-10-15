from typing import cast

import numpy as np


# Floyd's etc cycle-finding algos are useless here: the function is already fully tabulated so
# there's no saving space, and this is at worst `len(tree)-1` operations if there's no cycle.
def detect_cycle(
    heads: np.ndarray[tuple[int], np.dtype[np.intp]],
) -> np.ndarray[tuple[int], np.dtype[np.bool_]] | None:
    """Find a cycle of length > 1 in a directed graph where each node has an outdegree of 1 (maximal
    directed pseudoforest, functional graph).

    ## Input

    - `heads`: A 1d integer array such that `heads[i]` is the head of the `i`-th node.

    ## Output

    - `cycle`: A 1d bool arrays such that `cycle[j]` is true iff node `j` in the graph is in the
      cycle. If there is no cycle in the graph, this is `None`.
    """
    visited = np.zeros_like(heads, dtype=bool)
    # Next node to check
    pointer = 0
    # Current tour as an indicator array
    current_tour = np.empty_like(heads, dtype=bool)
    while True:
        # Skip until we find a node we yet have to visit
        while visited[pointer]:
            pointer += 1
            if pointer == len(heads):
                return None

        # We follow the out edges recursively, visiting nodes as we go until we get to a node that
        # we have already visited. At which point, if we have also visited this parent during the
        # current tour, we have found a cycle. Otherwise, it's verified and we can escape.

        current_tour.fill(False)
        parent = pointer
        while not visited[parent]:
            visited[parent] = True
            current_tour[parent] = True
            parent = heads[parent]
        # If we landed on a self-looping node, we're all good, otherwise we have found a cycle
        if parent != heads[parent] and current_tour[parent]:
            # Found a cycle! Reconstruct it.
            cycle_start = parent
            cycle_pointer = parent
            cycle = np.zeros_like(heads, dtype=bool)
            cycle[cycle_start] = True
            while (cycle_pointer := heads[cycle_pointer]) != cycle_start:
                cycle[cycle_pointer] = True
            return cycle


# TODO: split out a `contraction` function to make this more readable
#
# TODO: remove the constraint on the diagonal, it's annoying
def chu_liu_edmonds(
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Use the Chu‑Liu/Edmonds algorithm to find a maximum spanning arborescence from the weight
    matrix of a weighted directed graph rooted at the 0th node.

    ## Input

    - `scores`: A 2d numeric array such that `scores[i, j]` is the weight of the `$j→i$` edge (i.e.
      for `j` being the head of `i`) in the graph. It must not have any optimal self-loops, i.e. for
      all `i != j`, `scores[i, i] < scores[i, j]` (i.e. there is no optimal self-loops). An easy way
      to do that is to fill the diagonal of `scores` with `-np.inf`.

    ## Output

    - `heads` A 1d integer array such that `heads[i]` is the head of the `i`-th node in a maximum
      spanning arborescence of the graph rooted at `0`.
    """
    tree = cast(np.ndarray[tuple[int], np.dtype[np.intp]], np.argmax(scores, axis=1))
    # The root is self-looping
    tree[0] = 0
    cycle = detect_cycle(tree)

    if cycle is None:
        return tree
    else:
        # `cycle` is a boolean indicator array, so this holds the indices of the elements of the
        # cycle
        cycle_locs = cycle.nonzero()[0]
        cycle_heads = tree[cycle]
        cycle_scores = scores[cycle, cycle_heads]
        total_cycle_score = cycle_scores.sum()

        # Elements outside of the cycle
        noncycle = np.logical_not(cycle)
        noncycle_locs = noncycle.nonzero()[0]

        # scores of the cycle's potential heads
        metanode_head_scores = (
            scores[cycle, :][:, noncycle] - cycle_scores[:, np.newaxis] + total_cycle_score
        )
        if np.isinf(metanode_head_scores).all():
            raise ValueError("Score overflow: can't reliably find an arborescence.")

        # scores of the cycle's potential dependents
        metanode_dep_scores = scores[noncycle, :][:, cycle]
        # best noncycle head for each cycle dependent
        metanode_heads: np.ndarray[tuple[int], np.dtype[np.intp]] = np.argmax(
            metanode_head_scores, axis=0
        )
        # best cycle head for each noncycle dependent
        metanode_deps: np.ndarray[tuple[int], np.dtype[np.intp]] = np.argmax(
            metanode_dep_scores, axis=1
        )

        # scores of noncycle graph
        subscores = scores[noncycle, :][:, noncycle]
        # expand to make space for the metanode
        subscores = np.pad(subscores, ((0, 1), (0, 1)), "empty")
        # still no self-loop
        subscores[-1, -1] = -np.inf
        # set the contracted graph scores of cycle's potential heads
        subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(noncycle_locs.shape[0])]
        # set the contracted graph scores of cycle's potential dependents
        subscores[:-1, -1] = metanode_dep_scores[np.arange(noncycle_locs.shape[0]), metanode_deps]

        # MST with contraction; (n+1) in n+1
        contracted_tree = chu_liu_edmonds(subscores)

        # head of the cycle
        cycle_head = contracted_tree[-1]
        # fixed tree
        contracted_tree = contracted_tree[:-1]
        # initialize new tree; (t) in 0
        new_tree = np.full_like(tree, fill_value=-1)
        # fixed tree with no heads coming from the cycle: (n) in [0,1]
        contracted_subtree = contracted_tree < contracted_tree.shape[0]
        # add the nodes to the new tree
        new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[
            contracted_tree[contracted_subtree]
        ]
        # fixed tree with heads coming from the cycle
        contracted_subtree = np.logical_not(contracted_subtree)
        # add the nodes to the tree
        new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
        # add the old cycle to the tree
        new_tree[cycle_locs] = tree[cycle_locs]
        # root of the cycle
        cycle_root = metanode_heads[cycle_head]
        # add the root of the cycle to the new tree
        new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
        return new_tree


def _set_root(
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]], root: np.integer
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    """Force the `root`-th node to be the only node under the root by overwriting the weights of
    the other children of the root."""
    root_score = scores[root, 0]
    scores = scores.copy()
    scores[1:, 0] = -np.inf
    scores[root, 1:] = -np.inf
    scores[root, 0] = 0.0
    return scores, root_score


def maximum_spanning_dependency_tree(
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    in_place: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Use the Chu‑Liu/Edmonds algorithm (Chu and Liu, 1965 ; Edmonds, 1967) with the root
    pre-selecting variant described by Stanojević and Cohen's (2021) to find a maximum spanning
    dependency tree from the weight matrix of a weighted rooted directed graph.

    ## Input

    - `scores`: A 2d numeric array such that `scores[i, j]` is the weight of the `$j→i$` edge (i.e.
      for `j` being the head of `i`) in the graph and the 0-th node is the root. `scores[0, 0]` must
      be 0 and `scores[i, i]` must be -inf for i≠0.
    - `in_place`: if True, `scores` will be preserved, otherwise this functions might modify it
      in-place.

    ## Output

    - `tree`: A 1d integer array such that `tree[i]` is the head of the `i`-th node

    ## References

    - Chu, Yoeng-Jin, and Tseng-Hong Liu. 1965. On the Shortest Arborescence of a Directed Graph.
      Scientia Sinica 14: 1396–400.
    - Edmonds, Jack. 1967. Optimum Branchings. Journal of Research of the National Bureau of
      Standards 71B (4): 233. <https://doi.org/10.6028/jres.071b.032>.
    - Miloš Stanojević and Shay B. Cohen. 2021. [A Root of a Problem: Optimizing Single-Root
      Dependency Parsing](https://aclanthology.org/2021.emnlp-main.823/). In Proceedings of the 2021
      Conference on Empirical Methods in Natural Language Processing, pages 10540–10557, Online and
      Punta Cana, Dominican Republic. Association for Computational Linguistics.
    """

    scores = scores.astype(np.float64, copy=not in_place)
    # Cheap way to prevent self-loops
    np.fill_diagonal(scores, -np.inf)
    scores[0, 0] = 0.0

    tree = chu_liu_edmonds(scores)

    unconstrained_roots = np.argwhere(np.equal(tree[1:], 0)).reshape(-1) + 1
    if unconstrained_roots.shape[0] == 1:
        return tree

    # To find a dependency tree, we run our msa algorithm once per every unconstrained root, setting
    # that root as the only possible one.
    best_score, best_tree = -np.inf, None
    for root in unconstrained_roots:
        _scores, root_score = _set_root(scores, root)
        _tree = chu_liu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = tree_probs.sum() + root_score
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree

    assert best_tree is not None
    return best_tree

    return chu_liu_edmonds(scores)
