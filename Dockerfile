FROM debian:10
LABEL author='David Kauffman <dkauffman@stsci.edu>'

# Conda Setup Environment Variables
ENV DISPLAY=${ARG_DISPLAY}
ENV PYTHONDONTWRITEBYTECODE=1
ENV CONDA_ROOT=/opt/conda
ENV SKA_ENV=ska3
ENV PATH=${CONDA_ROOT}/env/${SKA_ENV}/bin:${CONDA_ROOT}/bin:${PATH}

ARG ARG_NAME_COLUMN
ARG ARG_TIME_COLUMN
ARG ARG_VALUE_COLUMN

# Special Test Case Variables
ENV NAME_COLUMN=${NAME_COLUMN}
ENV TIME_COLUMN=${TIME_COLUMN}
ENV VALUE_COLUMN=${VALUE_COLUMN}
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Qt5Agg
#

# Raven Variables
ENV RAVEN_SECRET_KEY=${ARG_RAVEN_SECRET_KEY}

# Archive Environment Variables
ENV ENG_ARCHIVE=/srv/telemetry/archive
ENV TELEMETRY_ARCHIVE="${ENG_ARCHIVE}/data/"
ENV STAGING_DIRECTORY="${ENG_ARCHIVE}/stage/"

# JETA Environment Variables
ENV JETA_SCRIPTS=/srv/jeta/code/scripts
ENV ARCHIVE_DEFINITION_SOURCE="${JETA_SCRIPTS}/sql/create.archive.meta.sql"

# Install core system packages
RUN set -x \
    && apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y \
        curl \
        supervisor \
        ipython \
        python3-matplotlib \
        libqt5gui5 \
        python3-pyqt4 \
        npm \
        nodejs \
    && apt-get clean


# Install required version of conda for the ska3 build script
RUN set -x \
    && curl -O https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh \
    && ls -la /opt \
    && bash ./Miniconda3-4.3.21-Linux-x86_64.sh -f -b -p ${CONDA_ROOT} \
    && rm -f Miniconda3-4.3.21-Linux-x86_64.sh \
    && echo 'export PATH='${CONDA_ROOT}'/bin:$PATH' >>/etc/profile

# Install ska3
# Warning: this URL is out of our control https://cxc.cfa.harvard.edu/mta/ASPECT/jska3-conda/linux-64/repodata.json
RUN set -x \
    && conda create -n ${SKA_ENV} -c https://cxc.cfa.harvard.edu/mta/ASPECT/jska3-conda --yes ska3-flight

# Create project directories
RUN set -x \
    && mkdir -p /srv/jeta/code \
    && mkdir -p /srv/jeta/log \
    && mkdir -p /srv/jeta/api \
    && mkdir -p /srv/jeta/jupyter

# Create dummy log file for testing
RUN touch /srv/jeta/log/tail.log;

# Copy source code into container
COPY jeta /srv/jeta/code/jeta
COPY raven /srv/jeta/api
COPY requirements /srv/jeta/requirements
COPY scripts ${JETA_SCRIPTS}

# Copy over setup.py for JETA
WORKDIR /srv/jeta/code
COPY setup.py /srv/jeta/code/setup.py

# ADD the start up script
ADD entrypoint.sh /entrypoint.sh
RUN set -x \
    && chmod +x /entrypoint.sh

# Set CWD to service root directory (useful with attaching shell)
WORKDIR /srv/jeta/

# Expose the port for raven
EXPOSE 9293

# jupyter notebook
EXPOSE 2150

# jupyterlab
EXPOSE 2151

# jupyterhub
EXPOSE 8080


ENTRYPOINT ["/entrypoint.sh"]