"""
Implement multiple methods of selecting the next site to toggle.

In order for a class to be considered a valid site selector, it must have the following methods

It must have a call operator that accepts an adjacency matrix of lattice.
Using an algorithm defined within the class, a single site will be selected for a spin flip.
"""

from .base import *
from .comb import *
