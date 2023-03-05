from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from hopsparser import mst


# Only positive weights becaux nx seems to have issues with them
@settings(deadline=1024)
@given(
    adjacency=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=64).map(lambda x: (x, x)),
        elements=st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            exclude_min=True,
            max_value=8.0,
            min_value=0.0,
        ),
    ),
)
def test_CLE(adjacency: NDArray):

    # Parsing setting: no self-loop, 0 is the root
    np.fill_diagonal(adjacency, -np.inf)
    adjacency[0] = -np.inf
    adjacency[0, 0] = 0.0

    # TODO: this could also test CLE one root if we use the big M trick
    graph = nx.from_numpy_array(adjacency.T, create_using=nx.DiGraph)
    edmonds = nx.algorithms.tree.branchings.Edmonds(graph)
    nx_arborescence = edmonds.find_optimum(kind="max", style="arborescence")
    nx_weight = sum(nx_arborescence.get_edge_data(*e)["weight"] for e in nx_arborescence.edges)

    mst_heads = mst.chuliu_edmonds(adjacency)
    mst_weight = adjacency[np.arange(len(adjacency)), mst_heads].sum()

    assert np.isclose(nx_weight, mst_weight)
