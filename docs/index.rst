DREiMac: Dimensionality Reduction with Eilenberg-MacLane Coordinates
====================================================================

DREiMac is a library for topological data coordinatization, visualization, and dimensionality reduction.
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, in the real and complex projective space, and in lens spaces.

In a few words, DREiMac takes as input a point cloud together with a topological feature of the point cloud (in the form of a persistent cohomology class), and returns a map from the point cloud to a well-understood topological space (a circle, a product of circles, a projective space, or a lens space), which preserves the given topological feature in a precise sense.
You can check the :ref:`theory section <theory>` for details and or the examples below to see how DREiMac works in practice.

Installing
----------

Make sure you are using Python 3.8 or newer.
DREiMac depends on the following python packages, which will be installed automatically when you install with pip:
`matplotlib`,
`numba`,
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
    notebooks/parameter_standard_range
    notebooks/parameters_prime_and_check_cocycle_condition

Further examples
----------------


.. toctree::
    :maxdepth: 2
    :caption: Circular coordinates

    notebooks/coil20

.. toctree::
    :maxdepth: 2
    :caption: Toroidal coordinates

    notebooks/bullseye
    notebooks/genusTwoSurface

.. toctree::
    :maxdepth: 2
    :caption: Real projective coordinates

    notebooks/ImagePatches

.. toctree::
    :maxdepth: 2
    :caption: Complex projective coordinates

    notebooks/twoSphere
    notebooks/movingDot

.. toctree::
    :maxdepth: 2
    :caption: Lens coordinates

    notebooks/circleLensCoordinates
    notebooks/MooreSpace

Authors
=======

Jose A. Perea, Luis Scoccola, Chris Tralie

