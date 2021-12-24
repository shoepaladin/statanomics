from setuptools import setup

setup(
    name='ATELibrary',
    version='0.0.5',

    url='https://github.com/shoepaladin/statanomics/causalmodels/ATE',
    author='Julian Hsu',
    author_email='tarobum27@gmail.com',
    license='MIT',
    py_modules=['ATELibrary'],
    install_requires=['numpy','statsmodels','scipy','econml],
)