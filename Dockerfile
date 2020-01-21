FROM ubuntu:latest
ENV PYTHONDONTWRITEBYTECODE=1
ENV CONDA_ROOT=/opt/conda
ENV SKA_ENV=ska3
ENV PATH=${CONDA_ROOT}/env/${SKA_ENV}/bin:${CONDA_ROOT}/bin:${PATH}
RUN set -x \
    && apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y \
        curl \
        supervisor \
    && apt-get clean
RUN set -x \
    && curl -O https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh \
    && ls -la /opt \
    && bash ./Miniconda3-4.3.21-Linux-x86_64.sh -f -b -p ${CONDA_ROOT} \
    && rm -f Miniconda3-4.3.21-Linux-x86_64.sh \
    && echo 'export PATH='${CONDA_ROOT}'/bin:$PATH' >>/etc/profile
RUN set -x \
    && conda create -n ${SKA_ENV} -c http://cxc.cfa.harvard.edu/mta/ASPECT/ska3-conda --yes ska3-flight

EXPOSE 9293

ENTRYPOINT ["/entrypoint.sh"]