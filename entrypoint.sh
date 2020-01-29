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

cd /srv/jeta/code/;
source activate ska3;
python setup.py install

# ---------------------------------------------------------------------------
# configure supervisor
#-------------------------------------------------------------------------------
cat <<END >| /etc/supervisord.conf
[supervisord]
nodaemon=true
logfile=/tmp/supervisord.log
pidfile=/tmp/supervisord.pid


#-------------------------------------------------------------------------------
# Cleanup stale Web API data and start supervisord
#-------------------------------------------------------------------------------
# rm -f /srv/api/raven/raven.pid
# python manage.py collectstatic --no-input --link --clear
if test -t 0; then
    /usr/bin/supervisord -c /etc/supervisord.conf &
    WAIT_PIDS=$!
    if [[ $@ ]]; then
        eval $@
    fi
    wait $WAIT_PIDS
else
    /usr/bin/supervisord -c /etc/supervisord.conf
fi

END

tail -f /dev/null