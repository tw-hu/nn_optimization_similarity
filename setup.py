from setuptools import find_packages, setup

setup(
    name='payload',
    description='custom scripts and tools for research',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)