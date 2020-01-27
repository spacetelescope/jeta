.. include:: references.rst

================================
Ska Analysis Tutorial
================================

Overview
--------

This document gives a tutorial introduction to analysis using the Ska analysis
environment, including basic configuration and a number
of examples.

The Ska analysis environment consists of a full-featured Python installation with
interactive analysis, plotting and numeric capability along with the Ska engineering
telemetry archive.

The telemetry archive consists of:

* Tools to ingest and compress telemetry from the CXC Chandra archive products.
* Compressed telemetry files in HDF5 format.  Each MSID has three associated products:

  - Full time-resolution data: time, value, quality
  - 5-minute statistics: min, max, mean, sampled value, number of samples
  - Daily statistics: min, max, mean, sampled value, standard deviation, percentiles (1,
    5, 16, 50, 84, 95, 99), number of samples.


Configure 
-----------------------------

The first requirement is to make sure that your local machine is set up to run
X11.  On a Microsoft windows PC that probably means having `Cygwin/X <http://x.cygwin.com/>`_
installed and running.  On a Mac you will need X11 installed.  If you don't
have this already the best option is to install X11.pkg from 
`<http://xquartz.macosforge.org/trac/wiki/Releases>`_.  From Linux you are
already set with X-windows.

To set up and to use the archive it is best to start with a "clean" environment
by opening a new X-terminal window.  It is assumed you are using ``csh`` or
``tcsh``.

The first step is to log in to a linux machine on either the HEAD LAN
(e.g. ``ccosmos``) or on the GRETA LAN (``chimchim``).  On the GRETA network
the Ska analysis environment is available from any GRETA machine, but the
64-bit machine ``chimchim`` will run analysis and telemetry access tasks much
faster.

To configure the Ska runtime environment and make a new xterm window with a black
background do the following::

  ssh -Y <username>@<hostname>  # where <username> and <hostname> are filled in
   <Enter password>

Initial file setup
~~~~~~~~~~~~~~~~~~~~~~~

First make a new xterm window, for instance with the following command::

  xterm -bg black -fg green &

Now focus on the new xterm window and set up your path and various environment
variables so all the tools are accessible and use the correct libraries.::

  source /proj/sot/ska/bin/ska_envs.csh

If you have not used Python and matplotlib before you should do the 
following setup::

  mkdir -p ~/.matplotlib
  cp $SKA/include/cfg/matplotlibrc ~/.matplotlib/
  
Next setup the interactive Python tool.  If you already have a ``~/.ipython``
directory then rename it for safekeeping::

  mv ~/.ipython ~/.ipython.bak

Now make a new default IPython profile and copy the Ska customizations::

  mkdir -p ~/.ipython
  ipython profile create --ipython-dir=~/.ipython
  cp $SKA/include/cfg/ipython_config.py ~/.ipython/profile_default/
  cp $SKA/include/cfg/10_define_impska.py ~/.ipython/profile_default/startup/
  
Finally it is quite useful to define aliases to get into one of the Ska
environments and adjust your prompt to indicate that you are using Ska.  The
command and file to modify depends on the shell you are using and the network.
First if you don't know what shell you are using then do::

  echo $0

This should say either ``csh`` or ``tcsh`` or ``bash``.

=================== =============  =============
Shell               Network        File
=================== =============  =============
``csh`` or ``tcsh`` HEAD           ~/.cshrc.user
``csh`` or ``tcsh`` GRETA          ~/.cshrc
``bash``            HEAD or GRETA  ~/.bashrc
=================== =============  =============

Now put the appropriate lines at the end of the indicated file::

  # Csh or tcsh
  alias ska  'unsetenv PERL5LIB; source /proj/sot/ska/bin/ska_envs.csh; set prompt="ska-$prompt:q"'
  alias skadev  'unsetenv PERL5LIB; source /proj/sot/ska/dev/bin/ska_envs.csh; set prompt="ska-dev-$prompt:q"'
  alias skatest  'unsetenv PERL5LIB; source /proj/sot/ska/test/bin/ska_envs.csh; set prompt="ska-test-$prompt:q"'
  alias pylab "ipython --pylab"

  # Bash
  alias ska='unset PERL5LIB; . /proj/sot/ska/bin/ska_envs.sh; export PS1="ska-$PS1"'
  alias skadev='unset PERL5LIB; . /proj/sot/ska/dev/bin/ska_envs.sh; export PS1="ska-dev-$PS1"'
  alias skatest='unset PERL5LIB; . /proj/sot/ska/test/bin/ska_envs.sh; export PS1="ska-test-$PS1"'
  alias pylab='ipython --pylab'

The ``ska``, ``skadev`` and ``skatest`` aliases are a one-way ticket.  Once you
get into a Ska environment the recommended way to get out or change to a
different version is to start a new window.

The ``pylab`` alias is just a quicker way to get into ``ipython --pylab``.

Ska, Ska-dev, and Ska-test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the GRETA network the Ska-test environment will typically
correspond to the Ska environment on the HEAD network.  The GRETA Ska
environment is tied to the FOT Matlab tools and may lag behind the
latest updates to the HEAD Ska environment.  

The Ska-dev environment is potentially unstable and should not normally be used
unless you are developing the Ska environment.

Basic Functionality Test
----------------------------------------------

Once you have done the configuration setup that was just described, open a new
xterm window and get into the Ska environment by use the ``ska`` or ``skatest`` alias::

  % ska

To test the basic functionality of your setup, try the following at the linux
shell prompt.  In this tutorial all shell commands are shown with a "% "
to indicate the shell prompt.  Your prompt may be different and you should
not include the "% " when copying the examples.::

  % ipython --pylab
  
You should see something that looks like::

    Python 2.7.1 (r271:86832, Feb  7 2011, 11:30:54) 
    Type "copyright", "credits" or "license" for more information.

    IPython 0.12 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    Welcome to pylab, a matplotlib-based Python environment [backend: Qt4Agg].
    For more information, type 'help(pylab)'.

    In [1]: 

Now read some data and make a simple plot by copying the following lines in ``ipython``::

  import Ska.engarchive.fetch as fetch
  tephin = fetch.MSID('tephin', '2009:001', '2009:002')
  plot(tephin.times, tephin.vals)
  
You should get a figure like:

.. image:: fetchplots/basic_func.png

Tools overview
----------------------------------------------
There are four key elements that are the basis for doing plotting and analysis
with the engineering archive.

* Ska.engarchive.fetch: module to read and manipulate telemetry data
* `IPython`_: interactive python interpreter
* `matplotlib`_: python plotting package with an interface similar to Matlab
* `NumPy`_: python numerical package for fast vector and array math

Ska.engarchive.fetch
~~~~~~~~~~~~~~~~~~~~~~
The tools to access and manipulate telemetry with the Ska engineering archive
are described in the :doc:`fetch_tutorial`.

IPython
~~~~~~~~~~
`IPython`_ is a command-line tool that provides a python prompt
that is the basis for interactive analysis.  At the core it provides
an interpreter for python language commands but with the addition of 
external packages like `matplotlib`_ and `numpy`_ it becomes a full-featured
data analysis environment.  `IPython`_ is similar in many ways to the command-line interface in Matlab or IDL.

**Links**

* `Main documentation page <http://ipython.org/ipython-doc/rel-0.12/interactive/index.html>`_
* `Tutorial (older but good) <http://onlamp.com/pub/a/python/2005/01/27/ipython.html>`_

.. include:: ipython_tutorial.rst

NumPy
~~~~~~~~
Even though it was not explicit we have already been using `NumPy`_ arrays in
the examples so far.  NumPy is a Python library for working with
multidimensional arrays. The main data type is an array. An array is a set of
elements, all of the same type, indexed by a vector of nonnegative integers.

.. include:: numpy_tutorial.rst

Matplotlib
~~~~~~~~~~~
.. include:: matplotlib_tutorial.rst

