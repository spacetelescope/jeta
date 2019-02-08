# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

from .version import package_version

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

# Write GIT revisions and SHA tag into <this_package/git_version.py>
# (same directory as version.py)
package_version.write_git_version_file()


setup(name='jeta',
      author='David Kauffman',
      description='Modules supporting jSka engineering telemetry archive',
      author_email='dkauffman@stsci.edu',
      entry_points={'console_scripts': ['jska_fetch = jeta.archive.get_telem:main']},
      py_modules=['jeta.archive.fetch', 'jeta.archive.converters', 'jeta.archive.utils',
                  'jeta.archive.get_telem'],
      version=package_version.version,
      zip_safe=False,
      packages=['jeta', 'jeta.archive.derived', 'jeta.tests'],
      package_dir={'jeta': 'jeta'},
      package_data={'jeta': ['*.dat', 'units_*.pkl', 'GIT_VERSION'],
                    'jeta.tests': ['*.dat']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
