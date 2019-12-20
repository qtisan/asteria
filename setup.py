from setuptools import setup, find_packages

setup(name='asteria',
      version='1.0',
      description='A project for some utility applications.',
      license="MIT",
      long_description='README.md',
      author='Lennon QIAN',
      author_email='qtisan@hotmail.com',
      url='https://github.com/qtisan/asteria.git',
      packages=find_packages(),
      scripts=['scripts/stocking', 'scripts/test'])
