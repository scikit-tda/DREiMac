DREiMac: Dimension Reduction with Eilenberg-MacLane Coordinates
===============================================================

DREiMac is a library for topological data coordinatization, visualization, and dimensionality reduction.
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, and in the real projective space.

Installing
----------

Make sure you are using Python 3.8 or 3.9.
DREiMac depends on the following python packages, which will be installed automatically when you install with pip:
`matplotlib`,
`numpy`,
`persim`,
`ripser`, and
`scipy`.

.. code-block::

    pip install dreimac


Details about the theory
------------------------

DREiMac is based on `cohomology <https://en.wikipedia.org/wiki/Cohomology>`_ and `Eilenberg-MacLane spaces <https://en.wikipedia.org/wiki/Eilenberg%E2%80%93MacLane_space#Bijection_between_homotopy_classes_of_maps_and_cohomology>`_, and turns persistent cohomology computations into topology-preserving coordinates for data.

For more details see the corresponding papers for for the circular coordinates algorithm [1]_, the toroidal coordinates algorithm [2]_, and the projective coordinates algorithm [3]_.

.. [1] *Sparse Circular Coordinates via Principal Z-bundles*. J.A. Perea. The Abel Symposium (Book Series): Topological Data Analysis, vol. 15, no.1, pp. 435-458, 2020

.. [2] *Toroidal Coordinates: Decorrelating Circular Coordinates With Lattice Reduction*. L. Scoccola, H. Gakhar, J. Bush, N. Schonsheck, T. Rask, L. Zhou, and J. A. Perea. 39th International Symposium on Computational Geometry, 2023

.. [3] *Multiscale Projective Coordinates via Persistent Cohomology of Sparse Filtrations*. J.A. Perea. Discrete Comput Geom 59, 175–225, 2018


Authors
-------

Jose A. Perea, Luis Scoccola, Chris Tralie


Contents
========


.. toctree::
    :maxdepth: 2
    :caption: Examples

    notebooks/coil20
    notebooks/bullseye
    notebooks/genusTwoSurface
    notebooks/ImagePatches



.. toctree::
    :maxdepth: 2
    :caption: Choosing parameters

    notebooks/parameters_n_landmarks_and_cocycle_idx
    notebooks/parameter_perc
    notebooks/parameters_prime_and_check_cocycle_condition


.. toctree::
    :maxdepth: 2
    :caption: API and FAQ

    api
    faq
