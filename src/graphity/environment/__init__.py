"""
Graphity spans two fields of physics. As such, it needs tools to model each field individually.

Graphity works on spin glasses with {-1,+1} spins.
We need simulators that time evolve these spin glasses both with and without metropolis-hastings acceptance.
Additionally, we need Hamiltonians which act on spin glasses to yield energies.
Lastly, we need tools to generate random spin glasses that we may examine.

Graphity also works on spacetime graphs.
These are adjacency matricies that whose entries take on the values {0, 1}.
Like with spin glasses, we need simulators, Hamiltonians, and generators for spacetime graph problems.
"""