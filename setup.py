from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A package to fit fragility functions.'

# Setting up
setup(
    name="pyFragility",
    version=VERSION,
    author="Laxman Dahal",
    author_email="<laxman.dahal@ucla.com>",
    url="https://github.com/laxmandahal/pyFragility",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="BSD 4-Clause License",
    tests_require=['pytest'],
    packages=find_packages(),
    platforms='any',
    install_requires=['scipy', 'matplotlib', 'numpy', 'statsmodels', 'sympy', 'numdifftools', 'pandas'],
    keywords=['python', 'fragility', 'fitting', 'function', 'collaps risk', 'qmle', 'mle'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
