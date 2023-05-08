# TODO: ideally, these auxiliary methods would be included in utils.py as a CombinatorialNumberSystem class.
# However, currently, numba's support for classes is experimental, so this more robust.
import numpy as np
from scipy.special import comb
from numba import jit


def combinatorial_number_system_table(maximum_number_vertices, maximum_dimension):
    lookup_table = np.zeros((maximum_number_vertices+1, maximum_dimension + 2), dtype=int)

    for i in range(maximum_number_vertices+1):
        for l in range(maximum_dimension + 2):
            lookup_table[i, l] = comb(i, l, exact=True)

    return lookup_table
    
def number_of_simplices_of_dimension(dimension, n_vertices, lookup_table):
    return lookup_table[n_vertices,dimension+1]

@jit
def combinatorial_number_system_forward(
    oriented_simplex: np.ndarray, lookup_table: np.ndarray
):
    dimension = len(oriented_simplex) - 1
    res = 0
    for l in range(dimension + 1):
        res += lookup_table[oriented_simplex[l], l + 1]
    return res


@jit
def combinatorial_number_system_d1_forward(v0: int, v1: int, lookup_table: np.ndarray):
    return lookup_table[v0, 1] + lookup_table[v1, 2]


@jit
def combinatorial_number_system_d2_forward(
    v0: int, v1: int, v2: int, lookup_table: np.ndarray
):
    return lookup_table[v0, 1] + lookup_table[v1, 2] + lookup_table[v2, 3]
