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
        'markdown',
        'six==1.16.0',
        'notebook==6.4.5',
        'nbconvert==6.4.4',
        'MarkupSafe==2.0.1',
        'ipympl==0.8.2',
        'ipywidgets==7.6.5',
        'ipykernel==6.5.0',
        'ipython==7.29.0',
        'ipython-genutils==0.2.0',
        'jupyter-client==7.0.6',
        'jupyter-core==4.9.1',
        'jupyter-telemetry==0.1.0',
        'Jinja2>3.0.2',
        'jsonschema==4.2.1',
        'ruamel.yaml==0.17.17',
        'Django==2.2.24',
        'djangorestframework==3.12.4',
        'djangorestframework-simplejwt==4.7.2',
        'Unipath==1.1',
        'gunicorn==20.1.0',
        'jhub-remote-user-authenticator==0.1.0',
        'jupyterhub-dummyauthenticator==0.3.1',
        'Cython==0.29.24',
        'numpy==1.20.3',
        'pandas==1.3.4',
        'astropy==4.3.1',
        'torch==1.10.0',
        'matplotlib==3.4.3',
        'seaborn==0.11.2',
        'plotly==5.3.1',
        'bokeh==2.3.3',
        'numba==0.54.1',
        'h5py==3.5.0',
        'tables==3.6.1',
        'pyyaks==3.3.3',
        'celery==5.2.2',
        'redis==4.1.0',
        'jupyterlab==3.2.2',
        'configurable-http-proxy==0.2.3',
        'jupyterhub-idle-culler',
        'sphinx_rtd_theme',
        'sphinxcontrib-versioning==2.2.1',
    ],
    scripts=[
        'scripts/sql/create.archive.meta.sql'
    ]
    # classifiers=[
    #     'License :: BSD 3-Clause',
    # ]
)
