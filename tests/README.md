# Tests
This folder mostly contains unit / integration tests for the graphity library.
Upon every commit, my servers run all the tests they find in this directory and notify me if any test fails unexpectedly.
This is known as continiuous integration.

The reason this process is helpful is that it helps localize broken commits.
Instead of having your code fail at some distant point in the future, far removed from the offending commit, with sufficient unit testing, breaking changes are detected in the commit that broke things.
Now, this doesn't catch all (or even many) bugs, because if your intution about the problem is wrong, you'll write bad unit tests.
How this is particularly helpful for us is that it ensure something that worked before and got a certain result doesn't magically break.

# Running Tests Manually
Open up a terminal at the root of the project and activate the virtual environment.
Then, run `pytest .`.
This make take 1-10 minutes, because it runs a full annealing cycle to check that the outputs of the annealing system still look sane.

# Adding Tests
If you add new features or functionality, you should also be adding new tests.
To add an entirely new file, pick a file name and prefix it with `test_`.
Inside the test file, prefix any unit test functions with a `test_` as well.
To understand the specifics of writing a unit test, see the PyTest documentation: https://docs.pytest.org/en/6.2.x/
You can also view some of the exisiting tests.