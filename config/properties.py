import os
from jeta.core.exceptions import ImproperlyConfigured

def get_env_variable(var_name):

    try:
        return os.environ[var_name]
    except:
        message = "Set the {} environment variable".format(var_name)
        raise ImproperlyConfigured(message)


INGEST_DIR=get_env_variable('JSKA_INGEST_DIR')
ARCHIVE_DIR=get_env_variable('JSKA_ARCHIVE_DIR')

NAME_COLUMN=get_env_variable('NAME_COLUMN')
TIME_COLUMN=get_env_variable('TIME_COLUMN')
VALUE_COLUMN=get_env_variable('VALUE_COLUMN')
