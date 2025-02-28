from distutils.core import setup

with open('requirements-lm-tools.txt') as fp:
    install_requires = fp.read()

setup(name='lm_experiments_tools',
      version='0.9.0',
      description='Tools for training language models with HF compatible interface.',
      author='Yura Kuratov',
      author_email='yurakuratov@gmail.com',
      packages=['lm_experiments_tools'],
      install_requires=install_requires
      )
