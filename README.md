# urPyPec 
This is a python port of Nicholgroup/urpec. https://github.com/nicholgroup/urpec
The goal of this port is to allow access to the Nichol group's proximity effect correction software without the need for a MATLAB license. 
I tried to design this translation to mirror the original as closely as possible (in terms of output, look, and types of functions used).
While I aimed for pythonic code wherever it was possible, I retained MATLABic code in some cases to allow easy conversion between changes or updates in the original and here. 
This program is not the most pythonic way to achieve urpec's output in python, but increasing the use and accessibility of the program should facilitate the expansion and improvement of free, open-source PEC software.

Compatibility should be added for multiple DXF layers as well as GDS files. Polygon fracturing into quadrilaterals and triangles from dose-corrected shapes should be added and their actual doses recalculated.


Please, let us know about any improvements to urpec.
For questions about the python translation, contact pbloom2@u.rochester.edu
For questions about the project as a whole and the MATLAB software, contact john.nichol@rochester.edu

By Paul Bloom, John Nichol, and Brian McIntyre
