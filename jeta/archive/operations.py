import os

import uuid
import pickle
import sqlite3

import numpy as np
import h5py
import tables

# import pyyaks.logger
# import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('ENG_ARCHIVE')
TELEMETRY_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')
# JETA_LOGS = get_env_variable('JETA_LOGS')


# logger = pyyaks.logger.get_logger(
#     filename=f'{JETA_LOGS}/jeta.operations.log',
#     name='jeta_operations_logger',
#     level='INFO',
#     format="%(asctime)s %(message)s"
# )

def _create_archive_database():
    """ Create an empty archive.meta.info.db3 database if it doesn't exist

        This file is responsible for tracking the ingest history/progess 
        as well as the individual files that have been ingested.
    """
    db_filepath = os.path.join(TELEMETRY_ARCHIVE,'archive.meta.info.db3')
    if not os.path.exists(db_filepath):
        with open(get_env_variable('JETA_ARCHIVE_DEFINITION_SOURCE'), 'r') as db_definition_file:
            db_definition_script = db_definition_file.read()
            print('Creating archive tracking database (sqlite3) {}'.format(db_filepath))
            db = sqlite3.connect(db_filepath)
            cur = db.cursor()
            cur.executescript(db_definition_script)
            cur.close()


def _create_root_content():
    """ Make empty files and directories for msids, msids.pickle

        msid directories: hold the index.h5, times.h5, and values.h5 files
        msid.pickle: a running list of msids encountered during the ingests.
    """

    empty = set()
    if not os.path.exists(f"{ENG_ARCHIVE}/logs"):
        os.makedirs(f"{ENG_ARCHIVE}/logs")

    if not os.path.exists(f"{ENG_ARCHIVE}/archive"):
        os.makedirs(f"{ENG_ARCHIVE}/archive")
    
    if not os.path.exists(f"{ENG_ARCHIVE}/staging"):
        os.makedirs(f"{ENG_ARCHIVE}/staging")

    if not os.path.exists(f"{TELEMETRY_ARCHIVE}/msids.pickle"):
        with open(f"{TELEMETRY_ARCHIVE}/msids.pickle", 'wb') as f:
            pickle.dump(empty, f, protocol=0)
    if not os.path.exists(f"{ENG_ARCHIVE}/processed_files"):
        os.makedirs(f"{ENG_ARCHIVE}/processed_files")
    
   
def _create_msid_index(msid):
    """Create and initialize the index file for an msid to hold a table of indices

    :param msid: the msid for which an index.h5 file will be created and initialized
    :type msid: str
    
    """
    with tables.open_file(
        f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/index.h5",
         driver="H5FD_CORE",
         mode='a'
        ) as idx_file:
        if idx_file.__contains__('/epoch') is False:
            compound_datatype = np.dtype([
                ('epoch', np.float64),
                ('index', np.uint64),
            ])
            table = idx_file.create_table(idx_file.root, 'epoch', compound_datatype) 


def _create_msid_dataset(msid, dtype, nrows, target, nbytes):
    """Create either the values.h5 or times.h5 for the passed msid

    :param msid: msid name as a string
    :param dtype: string representation of a numpy datatype (i.e. np.int32)
    :param nrows: the total number of rows estimated for the lifetime of the 
    :param target: times.h5 or values.h5
    :param nbytes: number of bytes for string values if dtype==str

    :returns: int: 0 if successful
    """
    
    h5shape = (0,)

    if dtype == 'str':
        dtype = h5py.string_dtype(encoding='utf-8', length=int(nbytes))
    h5type = tables.Atom.from_dtype(np.dtype(dtype))
    
    filters = tables.Filters(complevel=5, complib='zlib')
    
    with tables.open_file(f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/{target}.h5", 'a') as h5:
        h5.create_earray(
            h5.root,
            target,
            h5type,
            h5shape,
            title=msid,
            expectedrows=nrows,
            filters=filters
        )

    return 0


def _create_archive_files(msid):
    """Create the values.h5 and times.h5 for the lifetime of an msid

    :param msid: the msid that for which archive files are being created.
    :type msid: str
    :raises err: a generic `catch all` exception.
    """

    values_files = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/values.h5"
    times_files = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/times.h5"
    try:
        if not os.path.exists(values_files):
            with tables.open_file(
                    values_files,
                    mode='w'
                ) as values:
                values.close()
        if not os.path.exists(times_files):
            with tables.open_file(
                    times_files,
                    mode='w'
                ) as times:
                times.close()  
    except Exception as err:
        # TODO: Capture exception better
        print(err)
        if not os.path.exists(values_files):
            os.remove(values_files)
        if not os.path.exists(times_files):
            os.remove(times_files)
        raise err


def _create_msid_directory(msid):
    """Create the msid directory which will store all files associated with that msid

    :param msid: the msid for which a directory will be created
    :type msid: str
    """

    msid_directory_path = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/"
    if not os.path.exists(msid_directory_path):
        os.makedirs(msid_directory_path)


# def _reset_last_ingest_timestamp(msid, h5):
#     """ Reset the last ingest timestamp in the msid reference db
#     """
#     h5[msid].attrs['last_ingested_timestamp'] = 0


def calculate_expected_rows(sampling_rate):
    """Calculate the number of rows expected during the archive lifetime.

    :param sampling_rate: number of datapoints generated per second
    :type sampling_rate: int
    :return: the calculated description
    :rtype: int
    """
  
    ARCHIVE_LIFE = 10
  
    return sampling_rate * 60 * 60 * 24 * 365 * ARCHIVE_LIFE


def backup(msid='ALL', data_only=False):
    """Create a snapshot of the archive to restore.

    :param msid: msid archive to backup. Defaults to All msids, defaults to 'ALL'
    :type msid: str, optional
    :param data_only: only backup the index.h5, times.h5, and values.h5, defaults to False
    :type data_only: bool, optional
    """
    pass
 

def restore(uuid):
    """Restore the state of the archive to a particular point

    :param uuid: the uuid of the snapshot to restore
    :type uuid: uuid
    """


def truncate(filetype, date):
    """Truncate msid and statfiles for every archive file after date (to nearest
    year:doy)

    :param filetype: TBD
    :type filetype: null
    :param date: threshold data to truncate
    :type date: astropy.time.Time
    """
    pass

    # colnames = pickle.load(open(msid_files['colnames'].abs, 'rb'))

    # date = DateTime(date).date
    # year, doy = date[0:4], date[5:8]

    # # Setup db handle with autocommit=False so that error along the way aborts insert transactions
    # db = Ska.DBI.DBI(
    #     dbi='sqlite',
    #     server=msid_files['archfiles'].abs,
    #     autocommit=False
    # )

    # # Get the earliest row number from the archfiles table where year>=year and doy=>doy
    # out = db.fetchall('SELECT rowstart FROM archfiles '
    #                   'WHERE year>={0} AND doy>={1}'.format(year, doy))
    # if len(out) == 0:
    #     return
    # rowstart = out['rowstart'].min()
    # time0 = DateTime("{0}:{1}:00:00:00".format(year, doy)).secs

    # for colname in colnames:
    #     ft['msid'] = colname
    #     filename = msid_files['mnemonic_value'].abs # msid_files['msid'].abs
    #     if not os.path.exists(filename):
    #         raise IOError('MSID file {} not found'.format(filename))
    #     if not opt.dry_run:
    #         h5 = tables.open_file(filename, mode='a')
    #         h5.root.data.truncate(rowstart)
    #         h5.root.quality.truncate(rowstart)
    #         h5.close()
    #     logger.verbose('Removed rows from {0} for filetype {1}:{2}'.format(
    #         rowstart, filetype['content'], colname))

    #     # Delete the 5min and daily stats, with a little extra margin
    #     if colname not in fetch.IGNORE_COLNAMES:
    #         del_stats(colname, time0, '5min')
    #         del_stats(colname, time0, 'daily')

    # cmd = 'DELETE FROM archfiles WHERE (year>={0} AND doy>={1}) OR year>{0}'.format(year, doy, year)
    # if not opt.dry_run:
    #     db.execute(cmd)
    #     db.commit()
    # logger.verbose(cmd)


def destory(data_only=True):
    """Destory the archive by removing all data 

    :param data_only: if True only remove the data from the files., defaults to True
    :type data_only: bool, optional
    :return: a message annoucing the outcome of the operation.
    :rtype: str
    """
    # TODO: Add confirmation logic
    from shutil import rmtree
    if data_only:
        try:
            rmtree(ENG_ARCHIVE + '/archive/data/')
            return "Archive was destoryed."
        except FileNotFoundError as err:
            return "Nothing to do. Archive does not exist."


def add_msid_to_archive(msid, dtype, nrows, nbytes):
    """Add a single msid to the archive by creating the required files and directory structure

    :param msid: the msid to be added to the archive
    :type msid: str
    :param dtype: the numpy data type of the msid
    :type dtype: np.dtype
    :param nrows: the number of rows expected for the lifetime to the msid
    :type nrows: int
    :param nbytes: the number of bytes used in string representation (string msids only)
    :type nbytes: int
    """

    # Create the archive directory where the msid data will live
    _create_msid_directory(msid)
    
    # Create the values.h5, times.h5, and index.h5 for an msid
    _create_archive_files(msid)
    _create_msid_index(msid)

    # Create a the typed datasets
    _create_msid_dataset(msid, dtype, nrows, target='values', nbytes=nbytes)
    _create_msid_dataset(msid, 'float64', nrows, target='times', nbytes=None)


def initialize():
    """ Initialize the archive with all known msids

        This function creates and formats the persistent storage area 
        for each msids curated in the archive. 
    """
    _create_root_content()
    _create_archive_database()

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'a') as h5:
        for msid in h5.keys():
            add_msid_to_archive(
                msid, 
                dtype=h5[msid].attrs['numpy_datatype'].replace('np.', ''), 
                nrows=calculate_expected_rows(4),
                nbytes=h5[msid].attrs['nbytes']
            )
            h5[msid].attrs['last_ingested_timestamp'] = 0

if __name__ == "__main__":
    import jeta
    print(f"Initializing archive using jeta version {jeta.__version__} via cli")
    initialize()
