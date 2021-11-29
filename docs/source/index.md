# Getting Started with jeta

## What is jeta?

The `jeta` package is an API for monitoring and accessing FOS Engineering Telemetry Archive. The API exposes functions for ingesting/processing telemetry (tlm),
archive status/managment, and data fetching. Tools are built around the python data science stack (i.e. numpy, pandas, matplotlib ... etc.) 
and well as HDF5 standard and supporting python implementations (h5py, pytables). There are multiple ways to access and use the jeta package 
including directly after installing with `pip`, inside a pre-built enviroment with jupyter lab, a web API using HTTP request, and a web application
that provides a front-end for the most common use cases for the tools.

The package, `jeta` is a LITA functional-equivalent of [`Ska.eng_archive`](https://github.com/sot/eng_archive).
[Ska](https://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/pydocs/) is the "engineering telemetry archive is a suite of tools and data products" for
that supports the [Chandra](https://chandra.harvard.edu/about/spacecraft.html) X-Ray Observatiory. Where as `jeta` fulfills a similar role for JWST.
The package is still dependant on the Chandra package environment, [skare3](https://github.com/sot/skare3).

## Installing `jeta`

The jeta package can be installed and run locally against a local copy of the archive (WARNING. This is not recommend as data volumns will get extremely large. This feature is mainly for the purpose of development and testing). There are currently two methods of using `jeta` locally, first by installing directly on the system (or in a virtual envirnment) and second running a containerized version. In 
either case, a collection of environment variables must be set. A list of required environment variables and there descriptions
can be found in `<project_root>/scripts/build_env.bash`. You can run this script in a terminal in the following ways to set the 
environment variables automatically:

```bash
    source <project_root>/scripts/build_env.bash
    # or, and not the . prefix
    . <project_root>/scripts/build_env.bash
```

### Using setuptools
You can install the jeta package locally on your system or in a conda or virtualenv. First clone a copy of the source repo
and the use the `setup.py` module to install.

```bash
    git clone https://github.com/spacetelescope/jeta
    cd jeta
    # a list of python packages required to use jeta
    # are listed in the requirements/local.txt file
    python setup.py install
```
### Using docker

```bash
    # After setting the environment variables 
    # or creating a .env with them in the same dir
    # as the docker-compose.yml
    docker-compose -f docker-compose.yml build
    docker-compose -f docker-compose.yml up 
``` 

### Verify the Installation

```bash
    >>> import jeta
    >>> jeta.__version__
    '2.11.0'
```

## Using on in JupyterLab

Navigate to: https://jlab.stsci.edu/notebook/user/<user>/lab

From there you will be presented with your own area to run jupyter notebooks
with the `jeta` package already available.

## Basic Usage

```python
    from jeta.archive import fetch
    msid = fetch.MSID(MSID, tstart, tstop)
    print(msid.vals)
    print(msid.times)
    print(msid.means)
    print(msid.mins)
    print(msid.maxes)
```

```{toctree}
---
maxdepth: 2
caption: Contents
---
fetch.md
plotting.md
operations.md
ingest.md
update.md
```





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
