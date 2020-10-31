from distutils.core import setup

setup(name='graphity',
      version='0.1.0.3',
      description='Research in background independent quantum gravity, also known as quantum graphity.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      url='https://github.com/Matthew-McRaven/graphity',
      packages=['graphity',
      'graphity.agent',
      'graphity.environment', 
      'graphity.graph', 'graphity.grad',
      'graphity.nn',
      'graphity.replay',  
      'graphity.train'],
     )