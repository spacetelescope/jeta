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
    except Error as e:
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

    filenames = [ ntpath.basename(paths) for paths in sorted(glob.glob(f"{get_env_variable('STAGING_DIRECTORY')}*.h5"))]

    return filenames


def get_ingest_history():
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ingest_history")

    rows = cur.fetchall()
    return rows


def get_total_archive_area_size(PATH_VAR='TELEMETRY_ARCHIVE'):

    size_of_archive_in_bytes = 0

    for location, dirnames, filelist in os.walk(ENG_ARCHIVE):
        for file in filelist:
            file_path = os.path.join(location, file)
            size_of_archive_in_bytes += os.path.getsize(file_path)
    return size_of_archive_in_bytes


def is_in_archive(msid):

    with open(msid_files['colnames'].abs, 'rb') as f:
        colnames = list(pickle.load(f))
        return msid in colnames
