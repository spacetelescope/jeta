import os
import h5py
import tables
import uuid
import pickle
import numpy as np

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('ENG_ARCHIVE')
ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')
# ARCHIVE_METADATA = get_env_variable('ARCHIVE_METADATA')

msid_files = pyyaks.context.ContextDict('update.msid_files',
                                        basedir=ENG_ARCHIVE)
msid_files.update(file_defs.msid_files)


def _create_root_content():
    """ Make empty files and directories for msids, msids.pickle, and archive.meta.info.db3
    """

    empty = set()
    if not os.path.exists(f"{ENG_ARCHIVE}/msids.pickle"):
        with open(f"{ENG_ARCHIVE}/msids.pickle", 'wb') as f:
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


def backup():
    pass


def restore():
    pass


def truncate():
    pass


def destory(data_only=True):
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
    print(f"Initializing archive using jeta version {jeta.__version__}")
    initialize()
