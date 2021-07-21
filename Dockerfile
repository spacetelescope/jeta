FROM python:3.8-slim-buster

LABEL version="1.2.0"
LABEL maintainer="David Kauffman <dkauffman@stsci.edu>"
LABEL "edu.stsci"="Space Telescope Science Institute"

ARG ARG_NAME_COLUMN
ARG ARG_TIME_COLUMN
ARG ARG_VALUE_COLUMN
ARG ARG_RAVEN_SECRET_KEY

# Conda Setup Environment Variables
ENV DISPLAY=${ARG_DISPLAY}
ENV PYTHONDONTWRITEBYTECODE=1

ENV CONDA_ROOT=/opt/conda
ENV PATH=${CONDA_ROOT}/envs/${JETA_ENV}/bin:${CONDA_ROOT}/bin:${PATH}


ENV JETA_ENV=jeta

# Special Test Case Variables
ENV NAME_COLUMN=${NAME_COLUMN}
ENV TIME_COLUMN=${TIME_COLUMN}
ENV VALUE_COLUMN=${VALUE_COLUMN}
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Qt5Agg

# Raven Variables
ENV RAVEN_SECRET_KEY=${ARG_RAVEN_SECRET_KEY}

# Archive Environment Variables
ENV ENG_ARCHIVE=/srv/telemetry/
ENV TELEMETRY_ARCHIVE=/srv/telemetry/archive/
ENV STAGING_DIRECTORY=/srv/telemetry/staging/
ENV ALL_KNOWN_MSID_METAFILE=/srv/telemetry/archive/all_known_msid_sematics.h5

# Current status of the ingest pipeline
ENV INGEST_STATE="IDLE"

# UUID of the current ingest
ENV CURRENT_INGEST_ID="-1"

# JETA Environment Variables
ENV JETA_SCRIPTS=/srv/jeta/code/scripts
ENV JETA_ARCHIVE_DEFINITION_SOURCE="${JETA_SCRIPTS}/sql/create.archive.meta.sql"

# Install core system packages
RUN set -x \
    && apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y \
        curl \
        supervisor \
        ipython \
        python3-matplotlib \
        python3-pip \
        libqt5gui5 \
        python3-pyqt4 \
        wget \
        vim \
        dirmngr \ 
        apt-transport-https \
        lsb-release \ 
        ca-certificates \
        yarn \
        git-core \ 
        build-essential \ 
        openssl \
    && apt-get clean \
    && apt autoclean \
    && apt autoremove

# RUN set -x \
#     && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
#     && ls -la /opt \
#     && bash ./Miniconda3-py38_4.9.2-Linux-x86_64.sh  -f -b -p ${CONDA_ROOT} \
#     && rm -f Miniconda3-py38_4.9.2-Linux-x86_64.sh  \
#     && echo 'export PATH='${CONDA_ROOT}'/bin:$PATH' >>/etc/profile

# Create project directories
RUN set -x \
    && mkdir -p /srv/jeta/code \
    && mkdir -p /srv/jeta/log \
    && mkdir -p /srv/jeta/api

# Create JupyterHub JupyterLab Directories
# RUN set -x \
#         && mkdir -p /opt/jupyterhub/etc/jupyterhub/ \
#         && mkdir -p /opt/jupyterhub/etc/systemd \
#         && mkdir -p /srv/jeta/jupyter \
#         && mkdir -p /srv/jupyterhub

# Copy source code into container
COPY jeta /srv/jeta/code/jeta
COPY raven /srv/jeta/api
COPY requirements /srv/jeta/requirements
COPY scripts ${JETA_SCRIPTS}

# RUN set -x \
#     && conda config --env --set always_yes true \
#     && conda env create -n ${JETA_ENV} python=3.8.5

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
EXPOSE 9232

# jupyterhub
EXPOSE 5050

ENTRYPOINT ["/entrypoint.sh"]