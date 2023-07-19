---
title: 'DREiMac: Dimensionality Reduction with Eilenberg-MacLane Coordinates'
tags:
  - topological data analysis
  - unsupervised learning
  - dimensionality reduction
authors:
 - name: Jose A. Perea
   orcid: 0000-0002-6440-5096
   affiliation: 1
 - name: Luis Scoccola 
   orcid: 0000-0002-4862-722X
   affiliation: 1
 - name: Christopher J. Tralie
   affiliation: 2
affiliations:
 - name: Northeastern University
   index: 1
 - name: Ursinus College
   index: 2
date: 4 July 2023
bibliography: paper.bib
---

# Summary

DREiMac is a library for topological data coordinatization, visualization, and dimensionality reduction.
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, in the real and complex projective spaces, and in lens spaces.

In a few words, DREiMac takes as input a point cloud together with a topological feature of the point cloud (in the form of a persistent cohomology class), and returns a map from the point cloud to a well-understood topological space (a circle, a product of circles, a projective space, or a lens space), which preserves the given topological feature in a precise sense.

DREiMac is based on persistent cohomology [@persistent-cohomology], a method from topological data analysis; the theory behind DREiMac is developed in
[@circular-coords],
[@projective-coords],
[@lens-coords],
[@toroidal-coords].
DREiMac is implemented in Python, using Numba for the more expensive computations.
We test DREiMac periodically in Ubuntu, macOS, and Windows.

The documentation for DREiMac can be found [here](https://scikit-tda.org/DREiMac/index.html).


# Statement of need and main contributions

Topological coordinatization is witnessing increased application in domains such as
neuroscience [@rybakken-et-al] [@kang-xu-morozov] [@gardner-et-al],
dynamical systems [@vejdemo-pokorny-skraba-kragic],
and dimensionality reduction [@fibered].
The fast implementations and data science integrations provided in DREiMac are aimed at enabling other domain scientists in their pursuits.

To the best of our knowledge, the only publicly available software implementing cohomological coordinates based on persistent cohomology is Dionysus [@dionysus].
Dionysus is a general purpose library for topological data analysis, which in particular implements the original circular coordinates algorithm of [@desilva-morozov-vejdemo].

DREiMac adds to the current landscape of cohomological coordinates software by implementing various currently missing functionalities; we elaborate on these below.
DREiMac also includes functions for generating topologically interesting datasets for testing, various geometrical utilities including functions for manipulating the coordinates returned by the algorithms, and several example notebooks including notebooks illustrating the effect of each of the main parameters of the algorithms.

**Sparse algorithms.**
All of DREiMac's coordinates are sparse, meaning that persistent cohomology computations are carried on a simplicial complex built on a small sample of the full point cloud.
This gives a significant speedup when compared to algorithms which use a simplicial complex built on the entire dataset, since the persistent cohomology computation is the most computationally intensive part of the algorithm.
For a precise description of the notion of sparseness, see the papers that develop the algorithms that DREiMac implements 
[@circular-coords],
[@projective-coords],
[@lens-coords],
[@toroidal-coords].


**Improvements to the circular coordinates algorithm.**
DREiMac implements two new functionalities addressing two issues that can arise when computing circular coordinates for data.

The circular coordinates algorithm turns a cohomology class with coefficients in $\mathbb{Z}$ into a map into the circle.
However, since persistent cohomology is computed with coefficients in a field, the cohomology class is obtained by lifting a cohomology class with coefficients in $\mathbb{Z}/q\mathbb{Z}$, with $q$ a prime.
This lift can fail to be a cocycle, resulting in discontinuous coordinates, which are arguably not meaningful; see Figure \ref{figure:fix-cocycle} (right).
An algebraic procedure for fixing this issue is described in [@desilva-morozov-vejdemo], but has thus far not been implemented.
DREiMac implements this using integer linear programming.

![Parametrizing the circularity of a trefoil knot in 3D. Here we display a 2-dimensional representation, but the 3-dimensional point cloud does not have self intersections (in the sense that it is locally 1-dimensional everywhere). On the right, the output of the circular coordinates algorithm without applying the algebraic procedure to fix the lift of the cohomology class. On the left, the ouput of DREiMac, which implements this fix. Details about this example can be found in the documentation. \label{figure:fix-cocycle}](fix-cocycle.png){width=70%}

Another practical issue of the circular coordinates algorithm is its performance in the presence of more than one large scale circular feature (Figures \ref{figure:genus-two-toroidal} and \ref{figure:genus-two-circular}).
To address this, DREiMac implements the toroidal coordinates algorithm, introduced in [@toroidal-coords], which allows the user to select several 1-dimensional cohomology classes and returns coordinates that parametrize these circular features in a simpler fashion.

![Parametrizing the circularity of a surface of genus two in 3D. Here we display a 2-dimensional representation, but the 3-dimensional point cloud does not have self intersections (in the sense that it is locally 2-dimensional everywhere). This is DREiMac's output obtained by running the toroidal coordinates algorithm. The output of running the circular coordinates algorithm is in Figure \ref{figure:genus-two-circular}. Details about this example can be found in the documentation. \label{figure:genus-two-toroidal}](genus-2-toroidal-c.png){width=80%}


![Parametrizing the circularity of a surface of genus two in 3D. This output is obtained by running the circular coordinates algorithm. The parametrization obtained is arguably less interpretable than that obtained by the toroidal coordinates algorithm, shown in Figure \ref{figure:genus-two-toroidal}. \label{figure:genus-two-circular}](genus-2-circular-c.png){width=80%}


**Previously not implemented cohomological coordinates.**
DREiMac implements real projective, complex projective, and lens coordinates, introduced in [@projective-coords],
[@lens-coords].
These allow the user to construct topologically meaningful coordinates for point clouds using cohomology classes with coefficients in $\mathbb{Z}/2\mathbb{Z}$, $\mathbb{Z}$, and $\mathbb{Z}/q\mathbb{Z}$ ($q$ a prime), respectively, and in cohomological dimensions $1$, $2$, and $1$, respectively.


# Example

We illustrate DREiMac's capabilities by showing how it parametrizes the large scale circular features of the unprecessed COIL-20 dataset [@coil-20]; details about this example can be found in the documentation.
The dataset consists of gray-scale images of 5 objects, photographed from different angles.
As such, it consists of 5 clusters, each cluster exhibiting one large scale circular feature; see Figure \ref{figure:coil-20-pds}.

![Persistent cohomology of 5 clusters of unprocessed COIL-20 dataset. \label{figure:coil-20-pds}](coil-20-pds-h.png){width=100%}

We use single-linkage to cluster the data into 5 clusters and compute the persistent cohomology of each cluster.
We then run the circular coordinates algorithm on each cluster, using the most prominent cohomology class of each cluster.
We display the result in Figure \ref{figure:coil-20-res}.

![Unprocessed COIL-20 parametrized by clustering and circular coordinates. \label{figure:coil-20-res}](coil-20-res.png){width=95%}

# Acknowledgements

We thank Tom Mease for contributions and discussions.
J.A.P. and L.S. were partially supported by the National Science Foundation through grants CCF-2006661
and CAREER award DMS-1943758.


# References
