import os
import json
import glob
import ntpath

from operator import itemgetter

def get_env_variable(var_name):

    try:
        return os.environ[var_name]
    except:
        error_msg = 'Set the {} environment variable'.format(var_name)
        raise ValueError(error_msg)

def get_number_of_mnemoics_in_archive():

    mnemonic_count = 0

    for _, mnemonics, filenames  in os.walk(get_env_variable('TELEMETRY_ARCHIVE')):
        mnemonic_count += len(mnemonics)

    return mnemonic_count

def get_list_of_staged_files(include_path=False):

    filenames = [ ntpath.basename(paths) for paths in sorted(glob.glob(f"{staging_directory}*.h5"))]

    return filenames

def staging_area_status():

    supported_file_types = ['h5', 'CSV']
    staging_directory = get_env_variable('STAGING_DIRECTORY')

    status = {
        'files': None,
        'sizes': None,
        'max_size': None,
    }

    status['files'] = sorted(glob.glob(f"{staging_directory}*.h5"))

    status['sizes']  = [(name, os.path.getsize(name)) for name in status['files']]

    max_size_file = max(status['sizes'], key=itemgetter(1))

    status['max_size'] = {
        'file': max_size_file[0],
        'size_in_bytes': max_size_file[1],
    }

    return status

def get_total_archive_area_size(PATH_VAR='TELEMETRY_ARCHIVE'):

    archive_path=os.environ[PATH_VAR]

    size_of_archive_in_bytes = 0

    for location, dirnames, filelist in os.walk(archive_path):
        for file in filelist:
            file_path = os.path.join(location, file)
            size_of_archive_in_bytes += os.path.getsize(file_path)
    return size_of_archive_in_bytes
