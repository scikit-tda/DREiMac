# DREiMac: Dimension Reduction with Eilenberg-MacLane Coordinates

DREiMac is a library for topological data coordinatization, visualization, and dimensionality reduction.
Currently, DREiMac is able to find topology-preserving representations of point clouds taking values in the circle, in higher dimensional tori, and in the real projective space.

## Usage

Here is a simple example; please check the Jupyter notebooks in the `notebooks` directory for further examples.

```python
# basic imports
from dreimac import CircularCoords, GeometryExamples, PlotUtils
from persim import plot_diagrams

# 200 samples from a noisy circle
X = GeometryExamples.noisy_circle(200)
PlotUtils.plot_2d_scatter_with_different_colorings(X, point_size=10)

# the persistence diagram, showing a single prominent class
cc = CircularCoords(X, n_landmarks=200)
plot_diagrams(cc.dgms_)

# the data colored by the circle-valued map constructed by DREiMac
circular_coordinates = cc.get_coordinates()
PlotUtils.plot_2d_scatter_with_different_colorings(X, [circular_coordinates], point_size=10, cmap="hsv")
```

<img src="https://user-images.githubusercontent.com/1679929/230939790-c52e742e-14c4-41dc-9521-8336ff12a85a.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/1679929/230939814-24da9a0d-497f-474b-ac9a-8406a8ea2c58.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/1679929/230939822-67b76137-56d6-48d1-9e23-0fc86ecce1ef.png" width="250" height="250">


## Installing

Make sure you are using Python 3.
DREiMac depends on the following python packages, which will be installed automatically when you install with pip:
`matplotlib`,
`numba`,
`numpy`,
`persim`,
`ripser`, and
`scipy`.

~~~~~ bash
pip install dreimac
~~~~~

## Documentation and support

You can find the documentation at TODO.
If you have further questions, please [open an issue](https://github.com/scikit-tda/DREiMac/issues/new) [TODO] and we will do our best to help you.
Please include as much information as possible, including your system's information, warnings, logs, screenshots, and anything else you think may be of use.

## Running the tests

You can run the tests by running the following commands from the root directory of a clone of this repository.

```bash
pip install pytest
pip install -r requirements.txt
pytest .
```

## Details about the theory

DREiMac is based on [cohomology](https://en.wikipedia.org/wiki/Cohomology) and [Eilenberg-MacLane spaces](https://en.wikipedia.org/wiki/Eilenberg%E2%80%93MacLane_space#Bijection_between_homotopy_classes_of_maps_and_cohomology), and turns persistent cohomology computations into topology-preserving coordinates for data.

For more details see [[1]](#1) for the circular coordinates algorithm, [[2]](#2) for the toroidal coordinates algorithm, and [[3]](#3) for the projective coordinates algorithm.

## Authors

Chris Tralie, Tom Mease, Jose Perea, Luis Scoccola

## References

<a id="1">[1]</a> 
*Sparse Circular Coordinates via Principal Z-bundles*. J.A. Perea. The Abel Symposium (Book Series): Topological Data Analysis, vol. 15, no.1, pp. 435-458, 2020

<a id="2">[2]</a> 
*Toroidal Coordinates: Decorrelating Circular Coordinates With Lattice Reduction*. L. Scoccola, H. Gakhar, J. Bush, N. Schonsheck, T. Rask, L. Zhou, and J. A. Perea. 39th International Symposium on Computational Geometry, 2023

<a id="3">[3]</a> 
*Multiscale Projective Coordinates via Persistent Cohomology of Sparse Filtrations*. J.A. Perea. Discrete Comput Geom 59, 175–225, 2018

## License

This software is published under Apache License Version 2.0.
