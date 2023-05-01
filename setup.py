import sys
import os
import platform

from setuptools import setup

## Get version information from _version.py
import re
VERSIONFILE="dreimac/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Use README.md as the package long description  
with open('README.md') as f:
    long_description = f.read()


setup(
    name="dreimac",
    version=verstr,
    description="DREiMac: Dimension reduction with Eilenberg-MacLane coordinates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jose A. Perea, Luis Scoccola, Chris Tralie",
    author_email="ctralie@alumni.princeton.edu",
    license='Apache2',
    packages=['dreimac'],
    setup_requires=[
        'cython', 'numpy'
    ],
    install_requires=[
        'cython', 'numpy', 'scipy', 'matplotlib', 'persim', 'ripser', 'numba'
    ],
    extras_require={
        'testing': [ # `pip install -e ".[testing]"``
            'pytest'  
        ],
        'docs': [ # `pip install -e ".[docs]"`
            'sktda_docs_config'
        ],
        'examples': []
    },
    python_requires='==3.8,3.9',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    keywords='topological data analysis, dimension reduction'
)
