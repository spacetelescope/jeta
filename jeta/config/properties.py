import os
from jeta.core.exceptions import ImproperlyConfigured

def get_env_variable(var_name):

    try:
        return os.environ[var_name]
    except:
        message = "Set the {} environment variable".format(var_name)
        raise ImproperlyConfigured(message)


STAGING_DIRECTORY=get_env_variable('STAGING_DIRECTORY')

# FOF file columns from which to parse data.
NAME_COLUMN=get_env_variable('NAME_COLUMN')
TIME_COLUMN=get_env_variable('TIME_COLUMN')
VALUE_COLUMN=get_env_variable('VALUE_COLUMN')

lock_file_path = f'{get_env_variable("TELEMETRY_ARCHIVE")}/ingest.lock'
