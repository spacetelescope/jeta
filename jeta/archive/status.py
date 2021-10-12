import os
import pickle
import h5py
import sqlite3
import glob
import ntpath

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')

ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')

msid_files = pyyaks.context.ContextDict('update.msid_files',
                                        basedir=ENG_ARCHIVE)
msid_files.update(file_defs.msid_files)


def create_connection(db_file=msid_files['archfiles'].abs):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)

    return conn


def get_msid_count():
    """Get a count of the number of msids in msid reference file.

    :return: a count of the number of msids known to the system
    :rtype: int
    """
    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        return len(h5.keys())


def get_msid_names():
    """Get the authoritative list of msids known to the system

    :return: a list of msid
    :rtype: list
    """
    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        return sorted(list(h5.keys()))


def get_list_of_staged_files(include_path=False):
    """Returns a list of staged file records.

    The records are stored as tuples and contain:
    - the filename
    - the files size on disk
    - the ctime for the file
    - the start/stop coverage of the file

    :param include_path: [description], defaults to False
    :type include_path: bool, optional
    :return: a list of staged files as tuples
    :rtype: list
    """
    from pathlib import Path
    from jeta.staging.manage import get_file_coverage
    
    filenames = [(ntpath.basename(paths), Path(paths).stat().st_size, Path(paths).stat().st_ctime, *get_file_coverage(os.path.basename(paths))) for paths in sorted(glob.glob(f"{get_env_variable('STAGING_DIRECTORY')}/*.h5"))]

    return filenames


def get_list_of_files_in_range(tstart, tstop, target_dir=get_env_variable('STAGING_DIRECTORY')):
    """!!!Not Implemented!!!

    :param tstart: [description]
    :type tstart: [type]
    :param tstop: [description]
    :type tstop: [type]
    :param target_dir: [description], defaults to get_env_variable('STAGING_DIRECTORY')
    :type target_dir: [type], optional
    """
    pass


def get_current_ingest_id():
    """Get the uuid of the active ingest

    :return: the id of the active ingest
    :rtype: uuid
    """
    return os.getenv('JETA_CURRENT_INGEST_ID')


def get_ingest_state():
    """Get the state code of the active ingest

    :return: (STARTING, PREPROCESSING, PROCESSING, POSTPROCESSING, COMPLETE)
    :rtype: str
    """
    return os.getenv('JETA_INGEST_STATE')


def get_ingest_files(ingest_id):
    """Get a list of files ingested as part of a specific ingest operation

    :param ingest_id: uuid of an ingest operation
    :type ingest_id: uuid
    :return: a table representing ingest files linked to an ingest
    :rtype: list
    """
    conn = create_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM archfiles WHERE ingest_id={ingest_id};")

    rows = cur.fetchall()

    return rows


def get_ingest_history():
    """Get the full ingest history for the archive

    :return: a table representing ingest history
    :rtype: list
    """
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ingest_history")

    rows = cur.fetchall()
    return rows


def get_total_archive_area_size(area="archive"):
    """Get the size of a archive area in bytes

    :param area: the area in the archive to get for which to get the size, defaults to "archive"
    :type area: str, optional
    :return: size of the archive area in bytes
    :rtype: float
    """
    from pathlib import Path

    area_map = {
        'archive': Path('/srv/telemetry/archive/data/tlm'),
        'staging': Path(get_env_variable('STAGING_DIRECTORY')),
    }

    root_directory = area_map[area]

    return sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())


def is_in_archive(msid):
    """Checks if a msid has been ingested into the archive

    :param msid: the msid to check
    :type msid: str
    :return: True if an msid has had data recorded as ingested
    :rtype: bool
    """
    with open(msid_files['colnames'].abs, 'rb') as f:
        colnames = list(pickle.load(f))
        return msid in colnames
