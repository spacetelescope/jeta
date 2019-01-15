# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

from jSka.jeta.version import package_version

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

# Write GIT revisions and SHA tag into <this_package/git_version.py>
# (same directory as version.py)
package_version.write_git_version_file()


setup(name='jSka.jeta',
      author='David Kauffman',
      description='Modules supporting jSka engineering telemetry archive',
      author_email='dkauffman@stsci.edu',
      entry_points={'console_scripts': ['ska_fetch = Ska.engarchive.get_telem:main']},
      py_modules=['jSka.jeta.archive.fetch', 'Ska.jeta.archive.converters', 'Ska.jeta.archive.utils',
                  'Ska.jeta.archive.get_telem'],
      version=package_version.version,
      zip_safe=False,
      packages=['Ska', 'Ska.jeta', 'Ska.jeta.derived', 'Ska.jeta.tests'],
      package_dir={'Ska': 'Ska'},
      package_data={'Ska.jeta': ['*.dat', 'units_*.pkl', 'GIT_VERSION'],
                    'Ska.jeta.tests': ['*.dat']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
