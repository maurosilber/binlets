from setuptools import setup

setup(
    name='binlets',
    version='0.1.0',
    packages=['binlets'],
    url='https://github.com/maurosilber/binlet',
    license='MIT',
    author='Mauro Silberberg',
    author_email='maurosilber@gmail.com',
    description='Denoising via adaptive binning.',
    install_requires=['numpy', 'numba', 'scipy']
)
