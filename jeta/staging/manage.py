import os

import shutil
import h5py
import uuid

import pickle
import sqlite3
import glob
import ntpath

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
STAGING_DIRECTORY = '/Users/dkauffman/Projects/jSka/development_archive/stage/' # get_env_variable('STAGING_DIRECTORY')


# The backlog is a special activity that hosts ingest files that should still be ingested
# but processing them is behind. i.e. can be any number of files for any range.
BACKLOG_DIRECTORY = '/Users/dkauffman/Projects/jSka/development_archive/stage/backlog/'

msid_files = pyyaks.context.ContextDict('update.msid_files',
                                        basedir=ENG_ARCHIVE)
msid_files.update(file_defs.msid_files)


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


def get_archive_files(stage_dir=STAGING_DIRECTORY, format='h5'):
    """Get list of files from a staging directory
    """

    files = []
    files.extend(sorted(glob.glob(f"{stage_dir}E*.{str(format).upper()}.")))
    files.extend(sorted(glob.glob(f"{stage_dir}E*.{str(format).lower()}")))

    return files


def get_files_by_date(tstart, tstop, stage_dir=STAGING_DIRECTORY, format='h5'):
    files = _sort_ingest_files_by_start_time(get_archive_files(
            stage_dir,
            format
        )
    )
    return [
       file['filename'] for file in files if file['tstart'] <= tstart or tstop >= file['tstart']
    ]


def flag_activity_files_by_date(tstart, tstop):
    pass


def flag_activity_files_by_list(ingest_files=[]):
    pass


def get_files_for_activity(activity_id):
    pass


def add_ingest_file_to_activity(filename, src, dst):
    pass


def move_staged_files_to_activity(activity_dir=BACKLOG_DIRECTORY):
    ingest_files = get_archive_files
    for file in ingest_files:
        shutil.move(f"{file}", f"{activity_dir}{file['filename']}")


def restore_from_backlog_by_date(tstart, tstop):
    pass


def restore_from_backlog_by_list(ingest_files=[]):
    pass


def _create_activity_staging_area(name, description=""):

    _activity = f"{STAGING_DIRECTORY}{name}"
    if os.path.exists(_activity):
        raise IOError(f"Cannot create activity {_activity} already exists.")
    else:
        os.mkdir(_activity)
    return 0


def remove_activity(name):
    _activity = f"{STAGING_DIRECTORY}{name}"
    if os.path.exists(_activity):
        return os.remove(_activity)
