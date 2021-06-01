In order to make this test program work, you must first run `ray start --head`.

Ray is a framework that enables distributed computation.
This is particularly helpful for us because it allows us to run each lattice on a different computer, providing a near-linear speed-up.
Even if only running on a single computer, it is still necessary to start up Ray.

Please turn off ray after finishing any experiment.
This is down by executing `ray stop`.