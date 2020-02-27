#!/bin/bash

set -e


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

echo "INFO: initializing user packages..."

source activate ska3;

cd /srv/jeta/code/;
python setup.py install

cd /srv/jeta/api
pip install -r requirements.txt

cd /srv/jeta/requirements
pip install -r production.txt




# pip install jupyter

# set -x && pip install jupyterhub;
# set -x && pip install jhub_remote_user_authenticator;
# set -x && conda install -c conda-forge configurable-http-proxy;
# set -x && conda install notebook;
# set -x && conda install -c conda-forge jupyterlab;

# set -x && conda install ipywidgets;

# set -x && cd /srv/jupyterhub/config && yes Y | jupyterhub --generate-config



set -x && ln -snf /usr/share/fonts/truetype/dejavu /opt/conda/envs/ska3/lib/fonts;

echo "Finished setting up environment."

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

# [program:jupyter]
# directory=/srv/jeta/jupyter
# command=jupyter notebook --no-browser --ip=0.0.0.0 --port=2150 --allow-root --NotebookApp.token='' --NotebookApp.password=''
# stdout_logfile=/srv/jeta/log/jupyter.log
# stdout_logfile_maxbytes=0
# stderr_logfile=/srv/jeta/log/jupyter.err
# stderr_logfile_maxbytes=0

# [program:jupyterlab]
# directory=/srv/jeta/jupyter
# command=jupyter lab --no-browser --ip=0.0.0.0 --port=2151 --allow-root --NotebookApp.token='' --NotebookApp.password=''
# stdout_logfile=/srv/jeta/log/jupyterlab.log
# stdout_logfile_maxbytes=0
# stderr_logfile=/srv/jeta/log/jupyterlab.err
# stderr_logfile_maxbytes=0

# [program:jupyterhub]
# directory=/srv/jupyterhub
# command=jupyterhub -f /srv/jupyterhub/config/jupyterhub_config.py
# stdout_logfile=/srv/jeta/log/jupyterhub.log
# stdout_logfile_maxbytes=0
# stderr_logfile=/srv/jeta/log/jupyterhub.err
# stderr_logfile_maxbytes=0
# # WantedBy=multi-user.target


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

source deactivate

pip install jhub_remote_user_authenticator;
pip install notebook;
pip install jupyterlab;

#pip install jupyterhub-dummyauthenticator;

cd /srv/jupyterhub;
jupyterhub -f /srv/jupyterhub/config/jupyterhub_config.py;

tail -f /dev/null