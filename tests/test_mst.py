import math

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given, reproduce_failure, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hopsparser import mst


# Only positive weights becaux nx seems to have issues with negative ones. We also set a lower bound
# that's not too small, because nx overwrites (????) the original weigths in G with `weight +
# min_weight - 1 + (min_weight - max_weight)`, which clamps to 0 if the weight is something too
# small, like 2.225074e-308 or even 1e-16.
@settings(deadline=1024)
@given(
    adjacency=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=64).map(lambda x: (x, x)),
        elements=st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            max_value=8.0,
            min_value=2**-16,
        ),
    ),
)
def test_cle(adjacency: np.ndarray[tuple[int, int], np.dtype[np.float64]]):
    # Parsing setting: no self-loop, 0 is the root
    np.fill_diagonal(adjacency, -np.inf)
    adjacency[0, 1:] = -np.inf
    adjacency[0, 0] = 0.0

    # TODO: this could also test CLE one root if we use the big M trick
    graph = nx.from_numpy_array(adjacency.T, create_using=nx.DiGraph)
    # NetworkX isn't able to deal with infinite-weight edges anymore 🫠
    infinite_edges = [(u, v) for u, v, weight in graph.edges(data="weight") if math.isinf(weight)]
    graph.remove_edges_from(infinite_edges)

    nx_arborescence = nx.algorithms.tree.branchings.maximum_spanning_arborescence(
        graph,
        attr="weight",
        preserve_attrs=True,
    )
    nx_weight = np.sum([nx_arborescence.get_edge_data(*e)["weight"] for e in nx_arborescence.edges])

    mst_heads = mst.chuliu_edmonds(adjacency)
    mst_weight = adjacency[np.arange(len(adjacency)), mst_heads].sum()

    assert_allclose(nx_weight, mst_weight)


@settings(deadline=1024)
@given(
    adjacency=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=128).map(lambda x: (x, x)),
        elements=st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        ),
    ),
)
def test_cle_one_root(adjacency: np.ndarray[tuple[int, int], np.dtype[np.float64]]):
    # Parsing setting: no self-loop, 0 is the root
    np.fill_diagonal(adjacency, -np.inf)
    adjacency[0, 1:] = -np.inf
    adjacency[0, 0] = 0.0

    mst_heads = mst.chuliu_edmonds_one_root(adjacency)

    # No cycle
    assert mst.detect_cycle(mst_heads) is None
    # One root
    assert np.sum(mst_heads[1:] == 0) == 1
    # Just in case
    assert mst_heads[0] == 0
