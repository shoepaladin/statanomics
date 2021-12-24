from setuptools import setup

from my_pip_package import __version__

setup(
    name='my_pip_package',
    version='0.0.5',

    url='https://github.com/shoepaladin/statanomics/',
    author='Julian Hsu',
    author_email='tarobum27@gmail.com',
    license='MIT',
    py_modules=['my_pip_package'],
    install_requires=['numpy','statsmodels','scipy','econml'],
)