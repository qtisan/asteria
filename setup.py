from setuptools import setup

setup(name='asteria',
      version='1.0',
      description='A project for some utility applications.',
      license="MIT",
      long_description='README.md',
      author='Lennon QIAN',
      author_email='qtisan@hotmail.com',
      url='https://github.com/qtisan/asteria.git',
      packages=['asteria'],
      scripts=[
          'scripts/run',
          'scripts/test',
      ])
