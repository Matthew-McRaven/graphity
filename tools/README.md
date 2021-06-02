# Tools
## Geometric
This tool reports the minimum energy state it finds, and renders it to the screen. 
Visual inspection will indicate if it is geometric or not.
Future extensions will include code that indicates the degree of geometric-ness.

## Anneal
This tool sweeps over beta for a given N.
It simulates a set of lattices until equilibrium is achieved.
After equilibrium is achieved, observables like magnetization and specific heat are computed and graphed.
This tool performs simulation in a distributed fashion, so it requires some extra start-up steps.
These steps are described in the `anneal/README.md`.