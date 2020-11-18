from distutils.core import setup

setup(name='graphity',
      version='0.2.0',
      description='Research in background independent quantum gravity, also known as quantum graphity.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      url='https://github.com/Matthew-McRaven/graphity',
      packages=['graphity', 'librl',
      'graphity.agent', 'librl.agent',
      'graphity.environment', 
      'graphity.graph', 'graphity.grad',
      'librl.nn',
      'librl.replay',  
      'graphity.task', 'librl.task',
      'graphity.train']
     )