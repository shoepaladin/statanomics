from setuptools import setup

from my_pip_package import __version__

setup(
    name='my_pip_package',
    version=__version__,

    url='https://github.com/shoepaladin/statanomics/',
    author='Julian Hsu',
    author_email='tarobum27@gmail.com',

    py_modules=['my_pip_package'],
)