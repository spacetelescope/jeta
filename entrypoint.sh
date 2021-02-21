#!/bin/bash

set -e

set -x && alias jeta='python /srv/jeta/code/jeta/archive/update.py'

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
#     && conda config --env --set always_yes true \
#     && conda create -n ${SKA_ENV} -c https://cxc.cfa.harvard.edu/mta/ASPECT/jska3-conda --yes ska3-flight;

set -x && source activate ${SKA_ENV};

cd /srv/jeta/code/;
set -x && python setup.py install

cd /srv/jeta/api

cd /srv/jeta/requirements
pip install -r production.txt

cd /srv/jeta/api

set -x && python manage.py makemigrations && python manage.py migrate;

cat <<END | python manage.py shell
from django.contrib.auth.models import User
if not User.objects.filter(username='svc_jska').exists():
    User.objects.create_superuser('svc_jska', 'no-reply@stsci.edu', 'svc_jska')
END

set -x && conda install -c conda-forge configurable-http-proxy;
set -x && pip install jupyterhub==1.1.0
set -x && pip install 'jupyterlab<2.0'
set -x && jupyter labextension install -y @jupyterlab/hub-extension
set -x && jupyter labextension install @jupyter-widgets/jupyterlab-manager
set -x && jupyter serverextension enable --py jupyterlab --user

set -x && jupyter lab build

set -x && ln -snf /usr/share/fonts/truetype/dejavu /opt/conda/envs/jSka/lib/fonts;

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
jupyterhub -f /srv/jupyterhub/config/jupyterhub_config.py;

# Keep the container running
tail -f /dev/null
