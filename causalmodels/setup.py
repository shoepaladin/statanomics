from setuptools import setup

from my_pip_package import __version__

setup(
    name='my_pip_package',
    version='0.0.5',
    version=__version__,

    url='https://github.com/shoepaladin/statanomics/causalmodels',
    author='Julian Hsu',
    author_email='tarobum27@gmail.com',
    license='MIT',
    py_modules=['ATELibrary', 'HTELibrary'],
    install_requires=['numpy','statsmodels','scipy'],
)