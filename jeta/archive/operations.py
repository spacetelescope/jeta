import os
import h5py
import tables
import numpy as np

import pyyaks.logger
import pyyaks.context

import jeta.archive.file_defs as file_defs
from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
MSID_METAFILE = '/Users/dkauffman/System/Engineering/projects/fot/platform/notebooks/.meta/sematics.h5'


msid_files = pyyaks.context.ContextDict('update.msid_files',
                                        basedir=ENG_ARCHIVE)
msid_files.update(file_defs.msid_files)


def _create_msid_index(msid):
    with h5py.File(msid_files['mnemonic_index'].abs) as idx_file:
        if idx_file.__contains__('/epoch') is False:
            compound_datatype = np.dtype([
                ('epoch', np.float64),
                ('index', np.uint64),
            ])
            table = idx_file.create_table(idx_file.root, 'epoch', compound_datatype) 


def _create_msid_dataset(msid, dtype, expected_rows, target):
    
    h5shape = (0,)
    h5type = tables.Atom.from_dtype(np.dtype(dtype))
    filters = tables.Filters(complevel=5, complib='zlib')

    with h5py.File(msid_times_filepath) as h5_times:
        h5_times.create_earray(
            h5_times.root,
            target,
            h5type,
            h5shape,
            title=msid,
            expectedrows=expected_rows,
            filters=filters
        )

    return 0


def _create_archive_files(msid):
    if not os.path.exists(msid_files['mnemonic_value'].abs):
        try:
            values_h5 = tables.open_file(
                msid_files['mnemonic_value'].abs,
                mode='w'
            )
            times_h5 = tables.open_file(
                msid_files['mnemonic_times'].abs,
                mode='w'
            )
            times_h5.close()
            values_h5.close()
        except Exception as err:
            print(err)


def _create_msid_directory(msid):
        msid_directory_path = msid_files['msid'].abs
        if not os.path.exists(msid_directory_path):
            os.makedirs(msid_directory_path)


def add_msid_to_archive(msid, dtype, expected_rows):
    # Create the archive directory where the msid data will live
    _create_msid_directory(msid)
    
    # Create the values.h5, times.h5, and index.h5 for an msid
    _create_archive_files(msid)
    _create_msid_index(msid)

    # Create a the typed datasets
    _create_msid_dataset(msid, dtype, expected_rows, target='data')
    _create_msid_dataset(msid, 'float64', expected_rows, target='time')


def initialize():
    with h5py.File(MSID_METAFILE, 'r') as h5:
        for msid in h5.keys():
            print(msid)
            print(f"{msid} | {h5[msid].attrs['numpy_datatype'].replace('np.', '')}")


if __name__ == "__main__":
    initialize()