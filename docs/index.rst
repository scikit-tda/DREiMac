DREiMac: Dimension Reduction with Eilenberg-MacLane Coordinates
===============================================================

DREiMac is a library for topological data coordinatization, visualization, and dimensionality reduction.
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, and in the real projective space.

In a few words, DREiMac takes as input a point cloud together with a topological feature of the point cloud (in the form of a persistent cohomology class), and returns a map from the point cloud to a well-understood topological space (a circle, a product of circles, or a projective space), which preserves the given topological feature in a precise sense.
You can check the :ref:`theory section <theory>` for details and or the examples below to see how DREiMac works in practice.

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




Contents
========

.. toctree::
    :maxdepth: 2
    :caption: About

    theory
    api
    faq


.. toctree::
    :maxdepth: 2
    :caption: Choosing parameters

    notebooks/parameters_n_landmarks_and_cocycle_idx
    notebooks/parameter_perc
    notebooks/parameters_prime_and_check_cocycle_condition


.. toctree::
    :maxdepth: 2
    :caption: Examples

    notebooks/coil20
    notebooks/bullseye
    notebooks/genusTwoSurface
    notebooks/ImagePatches

Authors
=======

Jose A. Perea, Luis Scoccola, Chris Tralie

