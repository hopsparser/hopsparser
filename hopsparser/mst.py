from typing import Literal, cast

import numpy as np


# FIXME: we don't actually need this in CLE: we only need one critical cycle
def tarjan(tree: np.ndarray) -> list[np.ndarray]:
    """Use Tarjan's SCC algorithm to find cycles in a tree

    ## Input

    - `tree`: A 1d integer array such that `tree[i]` is the head of the `i`-th node

    ## Output

    - `cycles`: A list of 1d bool arrays such that `cycles[i][j]` is true iff the `j`-th node of
      `tree` is in the `i`-th cycle
    """
    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    # I think this is in a list to be able to mutate it in the closure, even though `nonlocal` exists
    _index = [0]
    cycles = []

    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]  # `_index` is of length 1 so this is also `_index[0]`???
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])  # type: ignore
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])  # type: ignore

        # There's a cycle!
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return

    # -------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles


# Floyd's etc cycle-finding algos are useless here: the function here is already fully tabulated so
# there's no saving space, and this is at worst `len(tree)-1` operations if there's no cycle.
def detect_cycle(
    heads: np.ndarray[tuple[int], np.dtype[np.intp]],
) -> np.ndarray[tuple[int], np.dtype[np.bool_]] | None:
    """Find a circle in a directed graph where each node has an outdegree of 1 (maximal directed
    pseudoforest, functional graph) and the 0-th node is self-looping.

    ## Input

    - `heads`: A 1d integer array such that `heads[i]` is the head of the `i`-th node. This assumes
      that `heads[0]` is set to `0`

    ## Output

    - `cycle`: A 1d bool arrays such that `cycle[j]` is true iff node `j` the graph is in the cycle
      If there is no cycle in the graph, this is `None`.
    """
    on_stack = np.ones_like(heads, dtype=bool)
    on_stack[0] = False
    pointer = 1
    while True:
        while not on_stack[pointer]:
            pointer += 1
            # We could stop one step before that if we know there are no self-loops but eh.
            if pointer == len(heads):
                return None
        current = np.zeros_like(heads, dtype=bool)
        parent = pointer
        while on_stack[parent]:
            on_stack[parent] = False
            current[parent] = True
            parent = heads[parent]
        if current[parent]:
            # Found a cycle!
            cycle_start = parent
            cycle_pointer = parent
            # Pyright isn't good at propagating shape typing yet
            cycle = cast(
                np.ndarray[tuple[int], np.dtype[np.bool_]], np.zeros_like(heads, dtype=bool)
            )
            cycle[cycle_start] = True
            while (cycle_pointer := heads[cycle_pointer]) != cycle_start:
                cycle[cycle_pointer] = True
            return cycle


# TODO: split out a `contraction` function to make this more readable
def chuliu_edmonds(
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Use the Chu‑Liu/Edmonds algorithm to find a maximum spanning arborescence from the weight
    matrix of a rooted weighted directed graph

    ## Input

    - `scores`: A 2d numeric array such that `scores[i][j]` is the weight of the `$j→i$` edge (i.e.
      for `j` being the head of `i`) in the graph and the 0-th node is the root.

    ## Output

    - `tree`: A 1d integer array such that `tree[i]` is the head of the `i`-th node
    """
    np.fill_diagonal(scores, -np.inf)  # prevent self-loops
    scores[0, 1:] = -np.inf
    scores[0, 0] = 0
    tree = cast(np.ndarray[tuple[int], np.dtype[np.intp]], np.argmax(scores, axis=1))
    cycle = detect_cycle(tree)
    if cycle is None:
        return tree
    else:
        # t = len(tree); c = len(cycle); n = len(noncycle)
        # indices of cycle in original tree; (c) in t
        cycle_locs = np.where(cycle)[0]
        # heads of cycle in original tree; (c) in t
        cycle_subtree = tree[cycle]
        # scores of cycle in original tree; (c) in R
        cycle_scores = scores[cycle, cycle_subtree]
        # total score of cycle; () in R
        total_cycle_score: np.ndarray[tuple[Literal[1]], np.dtype[np.floating]] = cycle_scores.sum()

        # locations of noncycle; (t) in [0,1]
        noncycle = np.logical_not(cycle)
        # indices of noncycle in original tree; (n) in t
        noncycle_locs = np.where(noncycle)[0]

        # scores of cycle's potential heads; (c x n) - (c) + () -> (n x c) in R
        metanode_head_scores = (
            scores[cycle][:, noncycle] - cycle_scores[:, np.newaxis] + total_cycle_score
        )
        if np.isinf(metanode_head_scores).any():
            raise ValueError("Score overflow: can't reliably find an arborescence.")
        # scores of cycle's potential dependents; (n x c) in R
        metanode_dep_scores = scores[noncycle][:, cycle]
        # best noncycle head for each cycle dependent; (n) in c
        metanode_heads: np.ndarray[tuple[int], np.dtype[np.intp]] = np.argmax(
            metanode_head_scores, axis=0
        )
        # best cycle head for each noncycle dependent; (n) in c
        metanode_deps: np.ndarray[tuple[int], np.dtype[np.intp]] = np.argmax(
            metanode_dep_scores, axis=1
        )

        # scores of noncycle graph; (n x n) in R
        subscores = scores[noncycle][:, noncycle]
        # expand to make space for the metanode (n+1 x n+1) in R
        subscores = np.pad(subscores, ((0, 1), (0, 1)), "constant")
        # set the contracted graph scores of cycle's potential heads; (c x n)[:, (n) in n] in R -> (n) in R
        subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
        # set the contracted graph scores of cycle's potential dependents; (n x c)[(n) in n] in R->
        # (n) in R
        subscores[:-1, -1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]

        # MST with contraction; (n+1) in n+1
        contracted_tree = chuliu_edmonds(subscores)
        # head of the cycle; () in n
        cycle_head = contracted_tree[-1]
        # fixed tree: (n) in n+1
        contracted_tree = contracted_tree[:-1]
        # initialize new tree; (t) in 0
        new_tree = np.full_like(tree, fill_value=-1)
        # fixed tree with no heads coming from the cycle: (n) in [0,1]
        contracted_subtree = contracted_tree < len(contracted_tree)
        # add the nodes to the new tree (t)[(n)[(n) in [0,1]] in t] in t = (n)[(n)[(n) in [0,1]] in n] in t
        new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[
            contracted_tree[contracted_subtree]
        ]
        # fixed tree with heads coming from the cycle: (n) in [0,1]
        contracted_subtree = np.logical_not(contracted_subtree)
        # add the nodes to the tree (t)[(n)[(n) in [0,1]] in t] in t = (c)[(n)[(n) in [0,1]] in c] in t
        new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
        # add the old cycle to the tree; (t)[(c) in t] in t = (t)[(c) in t] in t
        new_tree[cycle_locs] = tree[cycle_locs]
        # root of the cycle; (n)[() in n] in c = () in c
        cycle_root = metanode_heads[cycle_head]
        # add the root of the cycle to the new tree; (t)[(c)[() in c] in t] = (c)[() in c]
        new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
        return new_tree


def _set_root(
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]], root: int
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


def chuliu_edmonds_one_root(
    scores: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Repeatedly Use the Chu‑Liu/Edmonds algorithm to find a maximum spanning dependency tree from
    the weight matrix of a rooted weighted directed graph.

    **ATTENTION: this modifies `scores` in place.**

    ## Input

    - `scores`: A 2d numeric array such that `scores[i][j]` is the weight of the `$j→i$` edge (i.e.
      for `j` being the head of `i`) in the graph and the 0-th node is the root.

    ## Output

    - `tree`: A 1d integer array such that `tree[i]` is the head of the `i`-th node
    """

    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1
    if roots_to_try.shape[0] == 1:
        return tree

    # We find the maximum spanning dependency_tree by trying every possible root
    best_score, best_tree = None, None  # This is what's causing it to crash
    for root in roots_to_try:
        _scores, root_score = _set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = tree_probs.sum() + root_score
        if best_score is None:
            best_score = tree_score
            best_tree = _tree
        elif tree_score > best_score:
            best_score = tree_score
            best_tree = _tree

    assert best_tree is not None
    return best_tree
