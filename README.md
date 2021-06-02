# graphity
Research in background independent quantum gravity, also known as quantum graphity.

# Getting started.
This project requires python 3.8.
Newer versions of python will not work as of 2021/06/01.

After installing python 3.8, you will need to install pip (if you do not already have it).
Go to the https://pip.pypa.io/en/stable/installing/ for instructions.

After installing pip, we will create a virtual environment.
Virtual environments are a way of keep our dependencies seperate from those used by other project.
Failure to do this may mean that one project need version 2.0 of some lib, while we need 3.0.
This conflict would be impossible to resolve if we didn't create an isolated environment for just this project.

To install virtualenv, open a terminal/command prompt and type the command `python3.8 -m pip install virtualenv`.
Then, in the terminal, navigate to the directory which contains this README.
Execute the command `python3.8 -m virtualenv venv`.
After creating a virtual environment, we must activate it.
On Mac OS / Linux, this is down via the command `source venv/bin/activate`, assuming your command prompt is in the directory which contains this README.
On Windows, follow the directions here: https://docs.python.org/3/library/venv.html.

Now that we have an isolated environment for our dependencies, it's time to actually install the dependencies.
We do this by running the command `pip install -r requirements.txt`.
This will take several to many minutes depending on your internet speed.
Additionally, it will take about 80M of hard disk space.


# Source Code Layout
* `data/` Contains lists of graphs that can be used as ground states.
* `docs/` Contains code to generate pretty documentation for the Graphity package.
* `src/` Contains the graphity python package. Each directory has a README describing what each portion of the code does.
* `test/` Contains unit tests that provide a degree of confidence in our `graphity` package. This is a good place to look for examples on how to use the package.
* `tools/ ` Contains a grouping of useful scripts. For example, one tool before distributed annealing and generates mag/c/ms graphs for a given N.