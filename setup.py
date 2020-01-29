from setuptools import setup, find_packages

# with open("README.md") as f:
#     long_description = f.read()

setup(
    name='jeta',
    version='2.0.0-rc',
    description='Modules supporting JWST engineering telemetry archive.',
    license='MIT',
    long_description="",
    author='David Kauffman',
    author_email='dkauffman@stsci.edu',
    url='https://github.com/spacetelescope/jeta',
    packages=find_packages(include=[
        'jeta',
        'jeta.archive',
        'jeta.archive.derived',
        'jeta.config',
        'jeta.core',
        'jeta.ingest',
        'jeta.tests',
        ]),
    scripts=[
        'scripts/sql/create.archive.meta.sql'
    ],
    classifiers=[
        'Development Status :: v2.0.0 - Release Candidate',
        'License :: MIT license',
    ]
)
