# DREiMac
## Dimension Reduction with Eilenberg-MacClane Coordinates

# Python version

Code can be found in dreimac/.  To install, type

~~~~~ bash
python setup.py install
~~~~~

at the root of the directory.  Then, you can import dreimac from any python file or notebook

## Two Circles Example

The example below shows an example of the interactive circular coordinates interface on a point cloud of two noisy circles in 2D

<img src = "notesbooks/Selecting_Circles.gif">


## Torus Example

The example below shows an example of the interactive circular coordinates interface on a 3D point cloud of a torus

<img src = "notesbooks/Selecting_Torus.gif">



# Javascript version

Code can be found in dreimacjs/
CircluarCoords.html and ripser.html are the entry points

## Emscripten Compile options

~~~~~ bash
emcc --bind -s ALLOW_MEMORY_GROWTH=1 -O3 ripser.cpp
~~~~~

## MIME Types
* MIME Types for Javascript files should be text/javascript
* MIME Types for wasm files should be application/wasm
