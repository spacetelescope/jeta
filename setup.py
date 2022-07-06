from setuptools import setup, find_packages
from distutils.extension import Extension
import subprocess
import sys

from Cython.Build import cythonize
import numpy

# with open("README.md") as f:
#     long_description = f.read()

setup(
    name='jeta',
    description='Modules supporting JWST engineering telemetry archiving.',
    license='BSD 3-Clause',
    long_description='',
    long_description_content_type='text/markdown',
    author='David Kauffman',
    author_email='dkauffman@stsci.edu',
    url='https://github.com/spacetelescope/jeta',
    packages=find_packages(include=[
        'jeta',
        'jeta.archive',
        'jeta.archive.*',
        'jeta.staging',
        'jeta.archive.derived',
        'jeta.config',
        'jeta.core',
        'jeta.tests',
        ]),
    ext_modules = cythonize( 
        [
            Extension(
                "fastss", 
                ["jeta/archive/fastss.pyx"], 
                include_dirs=[numpy.get_include()]
            )
        ],
        compiler_directives={'language_level' : "3"}
    ),
    install_requires=[
        'Cython==0.29.24',
        'numpy==1.20.3',
        'pandas==1.3.4',
        'astropy==4.3.1',
        'torch==1.10.0',
        'matplotlib==3.4.3',
        'seaborn==0.11.2',
        'plotly==5.3.1',
        'numba==0.54.1',
        'h5py==3.5.0',
        'tables==3.6.1',
        'pyyaks==3.3.3',
    ],
    scripts=[
        'scripts/sql/create.archive.meta.sql'
    ]
    # classifiers=[
    #     'License :: BSD 3-Clause',
    # ]
)
