"""
Tools to work and operate on spacetime grpahs.

Graphity works on spacetime graphs with {0, 1} adjacency values.
Provides simulators that time-evolve spacetime graphs with metropolis-hastings acceptance.
Provides multiple Hamiltonians for assigning energies to spacetime graphs.
Provides methods for generating random adajcency matricies.
Creating ground state graphs and pure graphs is left as an exercise to the reader.
"""

from .generate import *
from .reward import *
from .sim import *