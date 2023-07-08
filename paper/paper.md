---
title: 'DREiMac: Dimension Reduction with Eilenberg-MacLane Coordinates'
tags:
  - topological data analysis
  - unsupervised learning
  - dimension reduction
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
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, in the real and complex projective space, and in lens spaces.

In a few words, DREiMac takes as input a point cloud together with a topological feature of the point cloud (in the form of a persistent cohomology class), and returns a map from the point cloud to a well-understood topological space (a circle, a product of circles, a projective space, or a lens space), which preserves the given topological feature in a precise sense.

DREiMac is based on persistent cohomology [@persistent-cohomology], a method from topological data analysis; the theory behind DREiMac is developed in
[@circular-coords],
[@projective-coords],
[@lens-coords],
[@toroidal-coords].
DREiMac is implemented in Python, using Numba for the more expensive computations.
We test DREiMac periodically in Ubuntu, macOS, and Windows.
(mention license?)

The documentation for DREiMac can be found [here](https://scikit-tda.org/DREiMac/index.html).


# Related work and statement of need

To the best of our knowledge, the only publicly available software implementing cohomological coordinates based on persistent cohomology is Dionysus [@dionysus].
Dionysus is a general purpose library for topological data analysis, which in particular implements the original circular coordinates algorithm of [@desilva-morozov-vejdemo].

DREiMac adds to the current landscape of cohomological coordinates software by implementing various currently missing functionalities, such as:
sparse algorithms;
toroidal, projective, and lens coordinates; 
(several example notebooks, and notebooks illustrating the effect of each of the main parameters)
(problem in lift of cocycles for circular coordinates, problem with decorrelating circular coordinates, add pictures for both)
(geometrical utilities, utilities for cohomological coordinates)
(example datasets)

All of DREiMac's coordinates are _sparse_, meaning that persistent cohomology computations are carried on a simplicial complex built on a small sample of the full point cloud.
This gives a significant speedup, since the persistent cohomology computation is the most computationally intensive part of the algorithm.

# Example

(COIL example, add axis to image (cluster + circular coordinate))

# Acknowledgements

We thank Tom Mease for contributions and discussions.
J.A.P. and L.S. were partially supported by the National Science Foundation through grants CCF-2006661
and CAREER award DMS-1943758.


# References
