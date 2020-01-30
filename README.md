# JWST Engineering Telemetry Archive (JETA)

A python package + Web API + scripting for getting data into and out of the JWST Engineering Telemetery Archive.

The package, `jeta` is a jSka functional-equivalent of [`Ska.eng_archive`](https://github.com/sot/eng_archive).
[Ska](https://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/pydocs/) is the "engineering telemetry archive is a suite of tools and data products" for
that supports the [Chandra](https://chandra.harvard.edu/about/spacecraft.html) X-Ray Observatiory. Where as `jeta` fulfills a similar role for JWST.
The package is still dependant on the Chandra package environment, [skare3](https://github.com/sot/skare3).

> NOTE: The project does not contain any telemetry data, just the tools.

## Getting Started

The project is built using Docker. To build the container, Docker 2.1.0.5 or greater will need to be installed on the
deployment or development machine.

Once you have Docker installed you running the container is a simple is running the commands in the
next section.

### Running The Container

```bash
    git clone https://github.com/spacetelescope/jeta.git
    cd jeta
    docker-compose run jska
```

### Prerequisites

All package dependencies and support scripts are build into the container.

### Environment Variables

For local development environment variables configured in a `.env` file. These same variables
need to be set inside the Docker container for running on server in test or production.

```bash
# Service-level Variables
WORKERS=<number of workers>

# Full path to the archive root on the real host or VM host
ENG_ARCHIVE=</path/to/archive/root>

# Archive Staging and Storage Directories
TELEMETRY_ARCHIVE=</path/to/ingested/data/root>
STAGING_DIRECTORY=</path/to/ingest-file/staging/area>

# Not relevant when running locally, but matters on a remote dev, test or prod server.
# Must be provided by sys/network admin. The default below is for Mac docker's defaults.
NETWORK_SUBNET=10.0.0.0/24
```

### Installing

The package is not intended to be installed as a stand-alone module, but instead be run and accessed via Docker.
However `jeta` can be installed using setuptools:

```bash
    # from the project root
    python setup.py install
```

## Running jeta with Celery

> NOTE: Requires additional install of [Redis](https://redis.io/) first to run.
> This will be a build more in the future.

### The worker to run tasks
celery -A jeta.ingest.controller worker --loglevel=info

### The default monitoring interface for tasks
celery -A jeta.ingest.controller flower --loglevel=info

> Initialize the archive
> This would normally be automated.

```bash
    # To initialize the creation of the telemetry archive.
    python run.py --create
```

## Running the tests

- TBD

## Coding Style

https://github.com/spacetelescope/style-guides

## Built With

* [Chandra Tools](https://cxc.harvard.edu/mta/ASPECT/tool_doc/pydocs/index.html) - Built around Chandra Tools
* [Django](https://docs.djangoproject.com/en/1.11/)- The web framework used is Django 1.11.x
* [Django Rest Framework](https://www.django-rest-framework.org/) - Restful API Framework


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [THELMA](https://github.com/ddkauffman/thelma).

## Acronyms

N/A

## Authors

* **David Kauffman** - *Initial work* - [David Kauffman](https://github.com/ddkauffman)

## Acknowledgments

* Amanda Arvai
* Tom Aldcroft
* Jean Connelly



```