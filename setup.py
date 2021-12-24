from setuptools import setup

from my_pip_package import __version__

setup(
    name='stnomics',
    version='0.0.5',
    url='https://github.com/shoepaladin/statanomics/',
    author='Julian Hsu',
    author_email='tarobum27@gmail.com',
    license='MIT',
    py_modules=['stnomics'],
    install_requires=['numpy','statsmodels','econml','scipy'],
)