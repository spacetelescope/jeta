from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from jeta.version import __version__

# with open("README.md") as f:
#     long_description = f.read()

setup(
    name='jeta',
    version=__version__,
    description='Modules supporting JWST engineering telemetry archiving.',
    license='BSD 3-Clause',
    long_description='',
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
    py_modules=['jeta.version'],
    scripts=[
        'scripts/sql/create.archive.meta.sql'
    ],
    classifiers=[
        f'Development Status :: v{__version__}',
        'License :: BSD 3-Clause',
    ]
)
