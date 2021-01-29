from distutils.core import setup
from setuptools import find_packages
import pkg_resources
install_reqs = []
install_reqs.extend(["more-itertools", "overrides", "future", "matplotlib", "scipy", "Pillow"])
install_reqs.extend(["torch ~= 1.7.0", "torchvision ~= 0.8.1", "pytest ~= 6.1"])
setup(name='graphity',
      version='0.3.0',
      description='Research in background independent quantum gravity, also known as quantum graphity.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      url='https://github.com/Matthew-McRaven/graphity',
      install_reqs=install_reqs,
      python_requires='~= 3.8',
      packages=find_packages('src'),
      package_dir={'': 'src'},
     )