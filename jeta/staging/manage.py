import os
import time
import shutil

import h5py

import pickle
import sqlite3
import glob
import ntpath
import uuid

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')


# The backlog is a special activity that hosts ingest files that should still be ingested
# but processing them is behind. i.e. can be any number of files for any range.
BACKLOG_DIRECTORY = f'{STAGING_DIRECTORY}backlog/'


def _format_activity_destination(dst):
    return f'{STAGING_DIRECTORY}{dst}/'


def _create_activity_staging_area(name, description=""):

    _activity = f"{STAGING_DIRECTORY}{name}"
    if os.path.exists(_activity):
        raise IOError(f"Cannot create activity {_activity} already exists.")
    else:
        os.mkdir(_activity)
    return 0


def _sort_ingest_files_by_start_time(list_of_files=[]):
    ingest_list = []

    for file in list_of_files:
        with h5py.File(file, 'r') as f:

            tstart = f['samples']["data1"].attrs['dataStartTime'][0]/1000
            tstop = f['samples'][f"data{len(f['samples'])}"].attrs['dataStopTime'][0]/1000
            ingest_list.append(
                {
                    'filename': f.filename,
                    'tstart': tstart,
                    'tstop': tstop,
                    'numPoints': f.attrs['/numPoints']
                }
            )

    return sorted(ingest_list, key=lambda k: k['tstart'])


def get_staged_files(stage_dir=STAGING_DIRECTORY, format='h5'):
    """Get list of files from a staging directory
    """

    files = []
    files.extend(sorted(glob.glob(f"{stage_dir}E*.{str(format).upper()}.")))
    files.extend(sorted(glob.glob(f"{stage_dir}E*.{str(format).lower()}")))

    return files


def get_staged_files_by_date(tstart, tstop, stage_dir=STAGING_DIRECTORY, format='h5'):
    files = _sort_ingest_files_by_start_time(get_staged_files(
            stage_dir,
            format
        )
    )
    return [
       file['filename'] for file in files if file['tstart'] <= tstart or tstop >= file['tstart']
    ]


def get_files_for_activity(name):
    _, _, filenames = next(os.walk(_format_activity_destination(name)))
    return filenames


def get_activity_file_count(name):
    # get_activity_file_count('') gets count of files in staging
    return len(get_staged_files(_format_activity_destination(name)))


def get_activity_count():
    _, dirnames, _ = next(os.walk(STAGING_DIRECTORY))
    return len(dirnames)


def get_list_of_activities():
    _, dirnames, _ = next(os.walk(STAGING_DIRECTORY))
    return dirnames


def get_file_coverage(ingest_filename):
    with h5py.File(f"{STAGING_DIRECTORY}/{ingest_filename}",  'r') as h5:
        tstart = h5['samples']["data1"].attrs['dataStartTime'][0]/1000
        tstop = h5['samples'][f"data{len(h5['samples'])}"].attrs['dataStopTime'][0]/1000
      
        return (tstart, tstop)


def flag_activity_files_by_date(tstart, tstop):
    # file root attr?
    pass


def flag_activity_files_by_list(ingest_files=[]):
    # file root attr?
    pass


def add_ingest_file_to_activity(filename, src, dst):
    try:
        shutil.move(f"{src}{filename}", f"{dst}{filename}")
    except Exception as err:
        raise err


def move_staged_files_to_activity(activity_dir=BACKLOG_DIRECTORY):
    ingest_files = get_staged_files()
    for file in ingest_files:
        shutil.move(f"{file}", f"{activity_dir}{file['filename']}")


def restore_from_backlog_by_date(tstart, tstop):
    pass


def restore_from_backlog_by_list(ingest_files=[]):
    pass


def restore_activity_to_staging(name):
    """ Move the files from an activity subdirectory back to the original
    staging directory.

        name: the name of an activity i.e. grouped data
        activity_path: full path of the activity
        filenames: list of filenames grouped under activity_path
    """
    try:
        activity_path = _format_activity_destination(name)
        _, _, filenames = next(os.walk(activity_path))
        if filenames != []:
            for f in filenames:
                shutil.move(
                    f"{activity_path}{f}",
                    # The string "{_format_activity_destination('')[:-1]}{f}
                    # should resolve to the current filename 'f' appended
                    # to the root staging path.
                    f"{_format_activity_destination('')[:-1]}{f}"
                )
    except Exception as err:
        raise err


def add_activity(name, ingest_files=None, description="", src=STAGING_DIRECTORY):
    """ Create a seperate space (a directory) for grouped data.
    """
    added_files = []

    logging_info = {
        'activity': name,
        'ingest_files': ingest_files,
        'added_files': added_files,
        'src': src,
        'timestamp': time.time(),
    }

    if ingest_files is None or ingest_files == []:
        _create_activity_staging_area(name, description="")
        return
    else:
        if _create_activity_staging_area(name, description="") == 0:
            for filename in ingest_files:
                # Apply an activity attribute to the hdf5 file?
                add_ingest_file_to_activity(
                    filename,
                    src=src,
                    dst=_format_activity_destination(name)
                    )
                added_files.append(filename)

    return logging_info


def remove_activity(name):
    _activity = f"{STAGING_DIRECTORY}{name}"
    if os.path.exists(_activity):
        return os.remove(_activity)
