import numpy as np
from dreimac.combinatorial import (
    combinatorial_number_system_table,
    combinatorial_number_system_forward,
    combinatorial_number_system_d2_forward,
)

class TestCombinatorialNumberSystem:

    def test_forward(self):

        lookup_table = combinatorial_number_system_table(5, 2)

        check = [
            [[0, 1, 2], 0],
            [[0, 1, 3], 1],
            [[0, 2, 3], 2],
            [[1, 2, 3], 3],
            [[0, 1, 4], 4],
        ]

        for simplex, answer in check:
            assert answer == combinatorial_number_system_forward(
                np.array(simplex), lookup_table
            )
            assert combinatorial_number_system_d2_forward(
                simplex[0], simplex[1], simplex[2], lookup_table
            ) == combinatorial_number_system_forward(np.array(simplex), lookup_table)
