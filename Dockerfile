FROM debian

# Conda Setup Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV CONDA_ROOT=/opt/conda
ENV SKA_ENV=ska3
ENV PATH=${CONDA_ROOT}/env/${SKA_ENV}/bin:${CONDA_ROOT}/bin:${PATH}

# Archive Environment Variables
ENV ENG_ARCHIVE=/srv/telemetry/archive
ENV TELEMETRY_ARCHIVE="${ENG_ARCHIVE}/data/"
ENV STAGING_DIRECTORY="${ENG_ARCHIVE}/stage/"

ENV JETA_SCRIPTS=/srv/jeta/code/scripts
ENV ARCHIVE_DEFINITION_SOURCE="${JETA_SCRIPTS}/sql/create.archive.meta.sql"

RUN set -x \
    && apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y \
        curl \
        supervisor \
        ipython \
    && apt-get clean

RUN set -x \
    && curl -O https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh \
    && ls -la /opt \
    && bash ./Miniconda3-4.3.21-Linux-x86_64.sh -f -b -p ${CONDA_ROOT} \
    && rm -f Miniconda3-4.3.21-Linux-x86_64.sh \
    && echo 'export PATH='${CONDA_ROOT}'/bin:$PATH' >>/etc/profile

RUN set -x \
    && conda create -n ${SKA_ENV} -c http://cxc.cfa.harvard.edu/mta/ASPECT/ska3-conda --yes ska3-flight


RUN set -x \
    && mkdir -p /srv/jeta/code \
    && mkdir -p /srv/jeta/log

RUN touch /srv/jeta/log/tail.log;

COPY jeta /srv/jeta/code/jeta
COPY scripts ${JETA_SCRIPTS}

WORKDIR /srv/jeta/code

COPY setup.py /srv/jeta/code/setup.py

WORKDIR /srv/jeta/code/

ADD entrypoint.sh /entrypoint.sh
RUN set -x \
    && chmod +x /entrypoint.sh

EXPOSE 9293

ENTRYPOINT ["/entrypoint.sh"]