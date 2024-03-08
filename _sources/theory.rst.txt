.. _theory:

Theory
======

DREiMac is based on `cohomology <https://en.wikipedia.org/wiki/Cohomology>`_ and `Eilenberg-MacLane spaces <https://en.wikipedia.org/wiki/Eilenberg%E2%80%93MacLane_space#Bijection_between_homotopy_classes_of_maps_and_cohomology>`_, and turns persistent cohomology computations into topology-preserving coordinates for data.
We now give details about what this means.
For more information, please see the corresponding papers for the circular coordinates algorithm [1]_, the toroidal coordinates algorithm [2]_, the projective coordinates algorithms [3]_, and the lens coordinate algorithm [4]_.

Inputs
------

DREiMac currently implements three cohomological coordinates procedures: the circular coordinates algorithm, the toroidal coordinates algorithm, and the real projective coordinates algorithm.
All procedures take 4 main parameters:

1. A dataset :code:`X`, which can be either a point cloud :math:`X \subseteq \mathbb{R}^n` or a distance matrix.

The dataset is interpreted as a finite sample of a continuous space :math:`X \subseteq M`.
In order to speed-up computations, a subsample of the initial dataset is computed :math:`L \subseteq X`, using a maxmin sampling procedure.

2. The :code:`n_landmarks` parameter determines the size of the subsample :math:`L`.

Then, a persistent cohomology computation is done using the Vietoris-Rips complex :math:`R(L)` of the metric space :math:`L`, which returns the persistence diagram of :math:`PH^n(R(L),\mathbb{Z}/p\mathbb{Z})`.
For this, a cohomological dimension :math:`n \in \mathbb{N}` and a prime :math:`p` must be chosen.
Right now, DREiMac supports arbitrary primes but only :math:`n=1`.
The choice of these parameters will depend on which of DREiMac's cohomological coordinates one is using.

- For circular and toroidal coordinates, the user can choose any prime :math:`p` and :math:`n` is set to :math:`1` automatically.

- For real projective coordinates, the prime :math:`p` is set to :math:`2` and :math:`n` is set to :math:`1`, automatically.

- For complex projective coordinates, the user can choose any prime :math:`p` and :math:`n` is set to :math:`2` automatically.

- For lens coordinates, the user can choose any prime :math:`p` and :math:`n` is set to :math:`1` automatically.

Then, looking at the persistence diagram, a cohomology class is chosen by the user, as follows.

3. The :code:`cocycle_idx` parameter is used to select a persistent cohomology class :math:`\eta \in PH^n(R(L);\mathbb{Z}/p\mathbb{Z})` by indicating its index when the persistent cohomology classes are ordered with respect to their persistence (i.e., their distance to the diagonal).
In the case of the toroidal coordinates algorithm, the corresponding parameter :code:`cocycle_idxs` specifies a list of persistent cohomology classes :math:`\eta_1 , \dots, \eta_\ell \in PH^1(R(L);\mathbb{Z}/p\mathbb{Z})`, such that the intersection of all of their supports is non-empty.

Finally:

4. The parameter :math:`0 <` :code:`perc` :math:`< 1` specifies the time in the filtration to construct the coordinates.


Ouput
-----

With these choices, DREiMac returns a continuous function

.. math::

   f_\eta \;:\; L^{\alpha}\; \xrightarrow{\;\;\;\;\;\;}\; K(G,n)

where:

:math:`K(G,n)` is an Eilenberg-MacLane space (EM space). Specifically, when using the

   - Circular coordinates algorithm, the EM space is the circle :math:`S^1`;
   - Toroidal coordinates algorithm with :math:`\ell` cohomology classes, the EM space is the :math:`\ell`-torus, i.e., the cartesian product of :math:`\ell` circles :math:`S^1 \times \dots \times S^1`;
   - Real projective coordinates algorithm, the EM space is a real projective space :math:`RP^k`. In this case, the dimension :math:`k` is selected using the :code:`proj_dim` parameter.
   - Complex projective coordinates algorithm, the EM space is a complex projective space :math:`CP^k`. In this case, the complex dimension :math:`k` is selected using the :code:`proj_dim` parameter. Note that in this case the real dimension will be :math:`2k`.
   - Lens coordinates algorithm, the EM space is a lens space. In this case, the dimension is selected using the :math:`k =` :code:`complex_dim`, which results in a lens space of real dimension :math:`2 k-1`. The prime determining the lens space is selected using the :code:`prime` parameter.

The space :math:`L^{\alpha}` is the :math:`\alpha`-thickening of :math:`L` in :math:`M`, defined as :math:`L^{\alpha} = \{m \in M : d_M(m,L) < \alpha\}`.

The number :math:`\alpha` is computed using the parameter :code:`perc` and the birth and death of the persistent cohomology class :math:`\eta` as follows: if the parameter :code:`standard_range` is :code:`True`, then

.. math:: \alpha = (1 - \rho)\cdot \max\{d_H^M(L,X), 2 \cdot birth(\eta)\} + \rho \cdot death(\eta),

where :math:`\rho =` :code:`perc`. If :code:`standard_range` is :code:`False`, then

.. math:: \alpha = (1 - \rho)\cdot \max\{d_H^M(L,X), birth(\eta)\} + \rho \cdot death(\eta).

In the case of the toroidal coordinates algorithm, one replaces :math:`birth(\eta)` with :math:`\max\{birth(\eta_i)\}` and :math:`death(\eta)` with :math:`\min\{death(\eta_i)\}`.

Guarantee
---------

When the parameter :code:`standard_range` is set to :code:`True`,
the output map :math:`f_\eta : L^{\alpha} \to K(G,n)` is guaranteed to preserve the cohomology class :math:`\eta`, in the following sense.
When one takes the pullback of the fundamental class of :math:`K(G,n)` along the map :math:`f_\eta`, one gets back a cohomology class corresponding to the user-chosen :math:`\eta`.


References
----------

.. [1] *Sparse Circular Coordinates via Principal Z-bundles*. J.A. Perea. The Abel Symposium (Book Series): Topological Data Analysis, vol. 15, no.1, pp. 435-458, 2020

.. [2] *Toroidal Coordinates: Decorrelating Circular Coordinates With Lattice Reduction*. L. Scoccola, H. Gakhar, J. Bush, N. Schonsheck, T. Rask, L. Zhou, and J. A. Perea. 39th International Symposium on Computational Geometry, 2023

.. [3] *Multiscale Projective Coordinates via Persistent Cohomology of Sparse Filtrations*. J.A. Perea. Discrete Comput Geom 59, 175–225, 2018

.. [4] *Coordinatizing Data With Lens Spaces and Persistent Cohomology*. L. Polanco, J.A. Perea. Proceedings of the 31 st Canadian Conference on Computational Geometry (CCCG), pp. 49-57, 2019




