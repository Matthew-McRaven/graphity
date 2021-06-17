""" 
Low-code methods for running spin-glass experiments.

##################
Pipelines
##################

The idea behind these pipelines is that they should more-or-less encapsulate the creation and management of an entire "thing".
For example, it should only take one line of code to kick off a distributed equalibiration experiment.
Pipelines acheve this by encapsulating all the necessary code within a single class.

Pipelines tend to come in two flavors:
 * Synchronous: These pipelines run on a single machine.
 * Synchronous Distributed: These pipelines implicitly use ray to parallelize computations when possible. This may possible involve multiple machines
If the name of a pipeline doesn't include an abbreviation of either, then by default it is synchronous.
"""
__all__ = ["graph, lattice"]
from .aug import * 
from .evolve import *
from .stat import *
from .autocor import *
import graphity.pipelines.graph
import graphity.pipelines.lattice
