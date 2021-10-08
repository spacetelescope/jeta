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
    """ Make empty files archive.meta.info.db3 if it doesn't exist
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
            print(err)
            if not os.path.exists(values_files):
                os.remove(values_files)
            if not os.path.exists(times_files):
                os.remove(times_files)
            raise err


def _create_msid_directory(msid):
        msid_directory_path = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/"
        if not os.path.exists(msid_directory_path):
            os.makedirs(msid_directory_path)


def calculate_expected_rows(sampling_rate):
    """ Calculate the number of rows expected during the archive lifetime.

        sampling_rate: number of datapoints generated per second
    """
    ARCHIVE_LIFE = 10
  
    return sampling_rate * 60 * 60 * 24 * 365 * ARCHIVE_LIFE


def backup(data_only=False):
    """[summary]

    Args:
        data_only (bool, optional): [description]. Defaults to False.
    """
    pass


def restore():
    """[summary]
    """


def truncate(filetype, date):
    """Truncate msid and statfiles for every archive file after date (to nearest
    year:doy)
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
    """Destory the archive by removing all data and

    Args:
        data_only (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    from shutil import rmtree
    if data_only:
        try:
            rmtree(ENG_ARCHIVE + '/archive/data/')
        except FileNotFoundError as err:
            return "Nothing to do. Archive does not exist."


def add_msid_to_archive(msid, dtype, nrows, nbytes):
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

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        for msid in h5.keys():
            add_msid_to_archive(
                msid, 
                dtype=h5[msid].attrs['numpy_datatype'].replace('np.', ''), 
                nrows=calculate_expected_rows(4),
                nbytes=h5[msid].attrs['nbytes']
            )

if __name__ == "__main__":
    import jeta
    print(f"Initializing archive using jeta version {jeta.__version__} via cli")
    initialize()
