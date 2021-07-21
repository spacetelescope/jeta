#!/bin/bash

set -e
set -o errexit
set -o pipefail
set -o nounset

#-------------------------------------------------------------------------------
# Handle shutdown/cleanup
#-------------------------------------------------------------------------------
function cleanup {
    if [ "Z$WAIT_PIDS" != "Z" ]; then
        kill -15 $WAIT_PIDS
    fi
}

# trap signals so we can shutdown sssd cleanly
trap cleanup HUP INT QUIT TERM

# Sync Docker home directories with Jupyterhub user spaces.
cd /home/
set -x && array=(*) && for dir in "${array[@]}"; do echo "Syncing for $dir"; id -u $dir &>/dev/null || useradd $dir; chown $dir:$dir $dir; done

# set -x \
#     && cd /srv/jeta/requirements \
#     && conda config --env --set always_yes true \
#     && conda env create -n ${SKA_ENV} -f jeta-conda.yml

# set -x && source activate ${JETA_ENV};

# Install the jeta tools inside the environment
cd /srv/jeta/code/
set -x && python setup.py install

# Install the API and Jupyterhub packages
# TODO: Isolate these services see branch LITA-35
# for a wip start on this effort.
cd /srv/jeta/requirements
pip install --upgrade pip
# pip install tld --ignore-installed six tornado --user
pip install -r production.txt

# Create the database for the API
cd /srv/jeta/api
set -x && python manage.py makemigrations && python manage.py migrate && python manage.py migrate authtoken;

# Create a default user for the API and generate a token
# FIXME: make p/w a parameter.
cat <<END | python manage.py shell
from django.contrib.auth.models import User
if not User.objects.filter(username='svc_thelma_api').exists():
    user = User.objects.create_superuser('svc_thelma_api', 'dkauffman@stsci.edu', 'svc_thelma_api')
    from rest_framework.authtoken.models import Token
    token = Token.objects.create(user=user)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print(token.key)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
END

# Install proxy tools for websockets
# set -x && conda install -c conda-forge configurable-http-proxy
# set -x && conda install -c conda-forge jupyterlab

# # Install Jupyterlab
# # set -x && pip install 'jupyterlab<2.0'
# set -x && jupyter labextension install -y @jupyterlab/hub-extension
# set -x && jupyter labextension install @jupyter-widgets/jupyterlab-manager
# set -x && jupyter serverextension enable --py jupyterlab --user

# # Build Jupyterlab
# set -x && jupyter lab build
# set -x && ln -snf /usr/share/fonts/truetype/dejavu /opt/conda/envs/jSka/lib/fonts;

# # ---------------------------------------------------------------------------
# # configure supervisor
# #-------------------------------------------------------------------------------
cat <<END >| /etc/supervisord.conf
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid

[program:raven]
directory=/srv/jeta/api
command=gunicorn --pid /srv/jeta/raven.pid --bind 0.0.0.0:9232 -w 2 -e DJANGO_SETTINGS_MODULE="config.settings.base" --access-logfile - --error-logfile - --log-level trace config.wsgi:application
stdout_logfile=/srv/jeta/log/raven.log
stdout_logfile_maxbytes=0
stderr_logfile=/srv/jeta/log/raven.err
stderr_logfile_maxbytes=0

END

#-------------------------------------------------------------------------------
# Cleanup stale Web API data and start supervisord
#-------------------------------------------------------------------------------
rm -f /srv/jeta/raven.pid
# python manage.py collectstatic --no-input --link --clear
if test -t 0; then
    /usr/bin/supervisord -c /etc/supervisord.conf &
    WAIT_PIDS=$!
    if [[ $@ ]]; then
        eval $@
    fi
    wait $WAIT_PIDS
else
    /usr/bin/supervisord -c /etc/supervisord.conf &
fi



# Start Jupyterhub with custom configuration
# jupyterhub -f /srv/jupyterhub/config/jupyterhub_config.py;

# Keep the container running
tail -f /dev/null
