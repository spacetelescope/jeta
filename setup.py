from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# with open("README.md") as f:
#     long_description = f.read()

setup(
    name='jeta',
    version='2.7.1',
    description='Modules supporting JWST engineering telemetry archiving.',
    license='BSD 3-Clause',
    long_description="",
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
    ext_modules = cythonize( [Extension("fastss", ["jeta/archive/fastss.pyx"], include_dirs=[numpy.get_include()])]),
    py_modules=['jeta.archive.version'],
    scripts=[
        'scripts/sql/create.archive.meta.sql'
    ],
    classifiers=[
        'Development Status :: v2.7.1',
        'License :: BSD 3-Clause',
    ]
)
