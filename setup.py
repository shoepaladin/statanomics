from setuptools import setup

setup(
    name='stnomics',
    version='0.0.8',
    url='https://github.com/shoepaladin/statanomics/',
    author='Julian Hsu',
    author_email='tarobum27@gmail.com',
    description='Installation of the Statanomics package. Thank you for using!',
    license='MIT',
    py_modules=['stnomics'],
    install_requires=['numpy','statsmodels','econml','scipy'],
)