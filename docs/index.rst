DREiMac: Dimension Reduction with Eilenberg-MacLane Coordinates
===============================================================

DREiMac is a library for topological data coordinatization, visualization, and dimensionality reduction.
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, and in the real projective space.


Installing
----------

Make sure you are using Python 3.
DREiMac depends on the following python packages, which will be installed automatically when you install with pip:
`matplotlib`,
`numpy`,
`persim`,
`ripser`, and
`scipy`.

.. code-block::

    pip install dreimac


Authors
-------

Chris Tralie, Tom Mease, Jose Perea, Luis Scoccola


Contents
--------


.. toctree::
    :maxdepth: 2

    api
    faq

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Examples 

    notebooks/coil20
    notebooks/bullseye
    notebooks/genusTwoSurface
    notebooks/ImagePatches

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Choosing parameters

    notebooks/parameters_n_landmarks_and_cocycle_idx
    notebooks/parameter_perc
    notebooks/parameters_prime_and_check_cocycle_condition


