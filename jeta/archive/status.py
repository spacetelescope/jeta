import os
import pickle
import sqlite3
import glob
import ntpath

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')

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

    with open(msid_files['colnames'].abs, 'rb') as f:
        colnames = pickle.load(f)
        return len(colnames)


def get_msid_names():

    with open(msid_files['colnames'].abs, 'rb') as f:
        colnames = pickle.load(f)
        return sorted(list(colnames))


def get_list_of_staged_files(include_path=False):
    from pathlib import Path
    from jeta.staging.manage import get_file_coverage
    
    filenames = [(ntpath.basename(paths), Path(paths).stat().st_size, Path(paths).stat().st_ctime, *get_file_coverage(os.path.basename(paths))) for paths in sorted(glob.glob(f"{get_env_variable('STAGING_DIRECTORY')}/*.h5"))]

    return filenames


def get_list_of_files_in_range(tstart, tstop, target_dir=get_env_variable('STAGING_DIRECTORY')):
    pass


def get_current_ingest_id():
    return os.getenv('JETA_CURRENT_INGEST_ID')


def get_ingest_state():
    return os.getenv('JETA_INGEST_STATE')


def get_ingest_files(ingest_id):

    conn = create_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM archfiles WHERE ingest_id={ingest_id};")

    rows = cur.fetchall()

    return rows


def get_ingest_history():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ingest_history")

    rows = cur.fetchall()
    return rows


def get_total_archive_area_size(area="archive"):

    from pathlib import Path

    area_map = {
        'archive': Path('/srv/telemetry/archive/data/tlm'),
        'staging': Path(get_env_variable('STAGING_DIRECTORY')),
    }

    root_directory = area_map[area]

    return sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())


def is_in_archive(msid):

    with open(msid_files['colnames'].abs, 'rb') as f:
        colnames = list(pickle.load(f))
        return msid in colnames
