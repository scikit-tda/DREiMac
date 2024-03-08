FAQ
===

.. contents::


Why does my output not look exactly like the output in the examples?
--------------------------------------------------------------------

There is more than one source of indeterminacy.
First, cohomological coordinates take values in spaces with many symmetries; for example, any rotation induces an isometry from the circle to the circle.
As such, cohomological coordinates are only well-defined up to rigid symmetries of the target space.
Secondly, since several of DREiMac's subroutines are numerical, the exact output can depend on your machine's specific architecture.

Having said this, if you are getting outputs that are qualitatively different from those in the examples, please let us know by filing an issue on DREiMac's repository.


How can I get further help?
---------------------------

If you have further questions, please 
`open an issue <https://github.com/scikit-tda/DREiMac/issues/new>`_ and we will do our best to help you.
Please include as much information as possible, including your system's information, warnings, logs, screenshots, and anything else you think may be of use.
