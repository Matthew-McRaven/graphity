"""
Implement multiple methods of temperature scheduling.

In order for a class to be considered a valid temperature scheduling object, it must have the following methods.

It must have a call operator that accepts a beta and delta_e. 
Using the knowledge of the current temperature and the change in energy from the last state, the object must return
a new (beta, probability of choosing that beta) pair.

It must also have a 0-arg reset() function.
reset() is called at the end of a number of sweeps to reset the internal state of the policy.
This is particularly important for non-adaptive policies such as simulated annealing, which have mutable internal state
that is used to generate the next beta.
"""
from .const import *
from .sa import *