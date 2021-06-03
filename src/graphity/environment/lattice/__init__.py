"""
Tools to work and operate on spin glass lattices.

Graphity works on spin glasses with {-1,+1} spins.
Provides simulators that time-evolve spin glasses with and without metropolis-hastings acceptance.
Provides multiple Hamiltonians for assigning energies to lattices.
Provides methods for generating random spin glasses.
Creating ground state spin glasses and pure glass-graphs is left as an exercise to the reader.
"""

from .generate import *
from .reward import *
from .sim import *