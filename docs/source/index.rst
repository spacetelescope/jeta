.. jSka documentation master file, created by
   sphinx-quickstart on Wed Oct 17 11:51:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started with jSka
=========================

Overview
--------
jSka is the runtime environment which the jSka suite of anomaly analysis tools operate within. The environment also include a variety
of ubiquitous Python Data Science Tools (numpy, Pandas, matplotlib ... etc.) and both jSka as well as the associated tools were created
using Python 3. The jSka-specific package available in the 2018.1.0.0 version of the jSka environment is the JWST Engineering Telemetry Archive (JETA) package.
JETA is responsible for ingesting Engineering Telemetry data into the archive. It is also the main interface for fetching data out of the archive or otherwise
interacting with archive.

Using IPython
-------------

1) Login to the Chandra Tool Server
2) Activate the jSka environment
3) Start IPython

.. code-block:: bash

    $ ssh server-name.stsci.edu
    (jSka) user@server-name$ source activate jSka
    (jSka) user@server-name$ ipython --matplotlib

Package versions can be verified as follows:

.. code-block:: ipython

    In [1]: import jeta

    In [2]: jeta.__version__
    Out[2]: '1.0'

Once IPython is started the fetch module can be imported as follows:

.. code-block:: ipython

    In [1]: from jeta.archive import fetch

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   fetch
   ingest
   infrastructure
   benchmarks
   code_style






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
