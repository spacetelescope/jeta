#!/usr/bin/env python
# see LICENSE.rst

from __future__ import print_function, division, absolute_import
from io import FileIO

import os
import glob
from uuid import uuid1
from functools import partial
import pickle
from random import seed
import torch

from collections import (
    Counter,
    OrderedDict,
    defaultdict,
    deque,
)

from astropy.time import Time

import pyyaks.logger
import pyyaks.context

import h5py
import tables
import numpy as np
import pandas as pd

import jeta.archive.fetch as fetch
import jeta.archive.file_defs as file_defs

from jeta.archive.utils import get_env_variable

# Frequency Per Day to ingest
INGEST_CADENCE = 2

# An assumption about the average number of files per ingest.
# default: 60 files covering an ~24mins interval each.
AVG_NUMBER_OF_FILES = 60

# The expected life time of the mission in years
MISSION_LIFE_IN_YEARS = 20

# The avg maximum number of rows per file per
# analysis performed by DMS.
MAX_ROWS_PER_FILE = 10_280_811

# Archive persistent storage locations on disk.
ENG_ARCHIVE = get_env_variable('ENG_ARCHIVE')
TELEMETRY_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')
JETA_LOGS = get_env_variable('JETA_LOGS')
ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')

# Calculate the number of files per year for archive space allocation prediction/allocation.
FILES_IN_A_YEAR = (AVG_NUMBER_OF_FILES * INGEST_CADENCE) * 365

ft = fetch.ft

msid_files = pyyaks.context.ContextDict('update.msid_files',
                                        basedir=ENG_ARCHIVE)
msid_files.update(file_defs.msid_files)
arch_files = pyyaks.context.ContextDict('update.arch_files',
                                        basedir=ENG_ARCHIVE)
arch_files.update(file_defs.arch_files)

logger = pyyaks.logger.get_logger(
    filename=f'{JETA_LOGS}/jeta.ingest.log',
    name='jeta_logger',
    level='INFO',
    format="%(asctime)s %(message)s"
)

# Temp global storage for ingest data
_values = None
_times = None
_epochs = None


def _reset_storage():
    global _values
    _values = defaultdict(partial(np.ndarray, 0))

    global _times
    _times = defaultdict(partial(np.ndarray, 0))

    global _epochs
    _epochs = defaultdict(float)


def _update_index_file(msid, epoch, index):
    """ Update the index file

        Parameters
        ----------
        filepath : str
            the index file path
        epoch : float
            the time
    """
    # filters = tables.Filters(complevel=5, complib='zlib')
    ft['msid'] = msid
    h5 = tables.open_file(
        f"{TELEMETRY_ARCHIVE}/data/tlm/{msid}/index.h5",
        driver="H5FD_CORE",
        mode="a"
    )

    if index is not None and epoch is not None:

        try:
            if h5.__contains__('/epoch') is False:
                compound_datatype = np.dtype([
                    ('epoch', np.float64),
                    ('index', np.uint64),
                ])
                table = h5.create_table(h5.root, 'epoch', compound_datatype)
            else:
                table = h5.root.epoch
            table.row['index'] = index
            table.row['epoch'] = epoch
            table.row.append()
            table.flush()
        except Exception as err:
            raise ValueError(f"Could not create epoch: {err}")
        finally:
            h5.close()


def _append_h5_col_tlm(msid, epoch):

    """Append new values to an HDF5 MSID data table.

    Parameters
    ----------
    msids : <class 'list'> of msids with data buffered for appending to the
            archive.
    """

    global _times
    global _values

    ft['msid'] = msid

    values_h5 = tables.open_file(
        f"{TELEMETRY_ARCHIVE}/data/tlm/{msid}/values.h5",
        mode='a'
    )

    if values_h5.__contains__('/values') is False:
        raise IOError(f"Archive file for {msid} value data does not contain root '/values'")

    times_h5 = tables.open_file(
        f"{TELEMETRY_ARCHIVE}/data/tlm/{msid}/times.h5",
        mode='a'
    )

    if times_h5.__contains__('/times') is False:
        raise IOError(f"Archive file for {msid} time data does not contain root '/times'")

    # Index should point to current number of rows
    index = values_h5.root.values.nrows

    try:
        values_h5.root.values.append(np.atleast_1d(_values[msid]))
        times_h5.root.times.append(np.atleast_1d(_times[msid]))
    except Exception as err:
        logger.error(f'{msid} couldnt append the normal way {type(_values[msid])} | {[_values[msid]]} | {_values[msid]}')


    _update_index_file(msid, epoch, index)

    values_h5.close()
    times_h5.close()

    return 0


def _is_file_already_in_db(ingest_file_path, db):

    # filename = os.path.basename(ingest_file_path)
    # if db.fetchall('SELECT filename FROM archfiles WHERE filename=?', (filename,)):
    #     logger.verbose('File %s already in archfiles - unlinking and skipping' % filename)
    #     os.unlink(ingest_file_path)
    return True


def calculate_delta_times(msid, times, epoch=None):

    if epoch is None:
        raise ValueError("Must have epoch")
    # epoch = Time(time.gmtime(0), format='unix').jd

    if times.ndim == 0:
        unix_times = [times]
    else:
        unix_times = [t/1000.0 for t in times]

    jd_times = Time(unix_times, format='unix').jd
    _times[msid] = np.diff(np.insert(jd_times, 0, epoch))


def sort_msid_data_by_time(msid, times=None, values=None, append=True):

    global _times
    global _values

    if append:
        _times[msid] = np.append(_times[msid], times)
        _values[msid] = np.append(_values[msid], values)

    msid_times = _times[msid]
    msid_values = _values[msid]

    idxs = torch.argsort(torch.from_numpy(msid_times))

    _times[msid] = np.array(msid_times[idxs])
    _values[msid] = np.array(msid_values[idxs])


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
    logger.info(
        (
            f"Data coverage for ALL ingest files discovered (tstart, tstop): "
            f"({Time(ingest_list[0]['tstart'], format='unix').iso},"
            f"{Time(ingest_list[-1]['tstop'], format='unix').iso})"
        )
    )
    return sorted(ingest_list, key=lambda k: k['tstart'])


def _auto_file_discovery(ingest_type, source_type):
    """ Automatically discover and return a list of files to ingest in the staging area.

    Parameters
    ==========
        ingest_type : str
            the type of file to match in the file search. 
            examples of valid values are: h5, hdf, csv, txt
        source_type : str
            the source type of the data. source_type of an empty string means sources of
            ingest_type.

    Returns
    =======
        A Python list of filenames based on the type of files to search for given by the
        discovery parameters.

    """
    logger.info(f"Attempting automatic file discovery in {STAGING_DIRECTORY} with ingest type {ingest_type}... ")

    ingest_files = []
    ingest_files.extend(sorted(glob.glob(f"{STAGING_DIRECTORY}/{source_type}*.{ingest_type}")))

    logger.info(f"{len(ingest_files)} file(s) staged in {STAGING_DIRECTORY} ...")

    return ingest_files


def _get_ingest_chunk_sequence(ingest_files):

    ingest_chunk_sequence = []

    tstop = ingest_files[0]['tstop']

    if len(ingest_files) == 0:
        chunk_len = 0
        return None
    if len(ingest_files) == 1:
        chunk_len = 1
        ingest_chunk_sequence = [1]
        return ingest_chunk_sequence
    else:
        chunk_len = 1
        for idx, f in enumerate(ingest_files[1:]):
            if f['tstart'] <= tstop or idx + 1 == len(ingest_files) - 1:
                chunk_len += 1
            else:
                ingest_chunk_sequence.append(chunk_len)
                chunk_len = 1
            tstop = f['tstop']
        ingest_chunk_sequence.append(chunk_len)
    return ingest_chunk_sequence


def move_archive_files(filetype, processed_ingest_files):

    if processed_ingest_files is not None:

        import tarfile
        import shutil

        os.chdir(STAGING_DIRECTORY)

        tarfile_name = f"stage_{int(Time(Time.now()).unix)}.tar"
        tar = tarfile.open(tarfile_name, mode='w')

        for ingest_file in processed_ingest_files:
            tar.add(ingest_file)
            os.remove(ingest_file)

        tar.close()
        shutil.move(
            tarfile_name,
            f"{msid_files['processed_files_directory'].abs}/{tarfile_name}"
        )


def _preprocess_hdf(ingest_files):
    ingest_files = _sort_ingest_files_by_start_time(ingest_files)

    # HDF5 files have been given with multiple identical start times
    # with different end times. Those files are grouped together here. 
    chunks = _get_ingest_chunk_sequence(ingest_files)

    return ingest_files, chunks


def _start_ingest_pipeline(ingest_type="csv", source_type='', provided_ingest_files=None, mdmap=None):
    """ This is the function internal to the system that archives data.

    Parameters
    ==========
    ingest_type : str
            this parameter impacts whichs filetype is gathered in a list to be ingested by the 
            automatic file discovery routine (i.e. jeta.archive.ingest.) .
    provided_ingest_files : list or list-like array of ingest files as string filenames.
            this is an optional parameter to supply a list of specific files to ingest.

    """

    # assign the list of files to ingest to `ingest_files` if no list is provided. 
    ingest_files = provided_ingest_files if provided_ingest_files is not None else _auto_file_discovery(ingest_type=ingest_type, source_type=source_type)

    # check if there were any ingest files
    if ingest_files:

        # Create a unique identifier for this ingest
        ingest_id = uuid1()

        # TODO: Write to database that an ingest was started.

        if ingest_type == 'csv' or ingest_type == 'CSV':
            _process_csv(ingest_files, ingest_id=ingest_id, single_msid=False)
            
        elif ingest_type == 'h5':
            # Preprocess HDF5 files to match DF interface
            logger.info('Starting HDF5 file pre-processing ...')
            ingest_files, chunks = _preprocess_hdf(ingest_files)
            logger.info('Completed HDF5 file pre-processing ...')

            logger.info('Starting HDF5 file data ingest ...')
            # out = db.fetchone('SELECT count(*) FROM ingest_history')
            processed_ingest_files = _process_hdf(
                ingest_files,
                ingest_files[0]['tstart'],
                ingest_files[-1]['tstop'],
                ingest_id=ingest_id.int,
                chunks=chunks,
                mdmap=mdmap
            )
            logger.info('Completed HDF5 file data ingest ALL data ...') 
        else:
            raise ValueError('Ingest type parameter is invalid. Valid options are csv or h5.')
        logger.info('Moving HDF5 ingest file to tmp storage ...')  
        # move_archive_files(filetype, processed_ingest_files)
    else:
        logger.info('No ingest files discovered in {STAGING_DIRECTORY}')


def _process_csv(ingest_files, ingest_id, single_msid=False):
    
    processed_files = []

    if single_msid == True:
        for f in ingest_files:
            fof = pd.read_csv(f) 
            
            _reset_storage()
            msid = fof['Telemetry Mnemonic'][0]
        
            with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
                npt = h5[msid].attrs['numpy_datatype'].replace('np.', '')
                times = fof['Observatory Time'].str.replace('/', '-')
                times = Time(times.to_list() , format='iso').jd
                
                sort_msid_data_by_time(msid, times, fof['EU Value'].to_list())
                _values[msid] = _values[msid].astype(npt)
                # Store times as deltas instead of as unix timestamps
                # _times[msid] = np.diff(np.insert(_times[msid], 0, _times[msid][0])) 
            
                _append_h5_col_tlm(msid=msid, epoch=_times[msid][0])

                processed_files.append(f)
           

    if single_msid == False:
        print("Processing FOF")
        with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
            for f in ingest_files:
                _reset_storage()
                fof = pd.read_csv(f)
                msids = fof['Telemetry Mnemonic'].unique().tolist()
                print(msids)
                for m in msids:
                    npt = h5[m].attrs['numpy_datatype'].replace('np.', '')
                    times = fof.loc[fof['Telemetry Mnemonic']==m, ['Observatory Time']]
                    times = times['Observatory Time'].str.replace('/', '-')
                    times = Time(times.tolist() , format='iso').jd
                    values = fof.loc[fof['Telemetry Mnemonic']==m, ['EU Value']]
                    sort_msid_data_by_time(m, times, values.values.tolist())
                    _values[m] = _values[m].astype(npt)
                    _append_h5_col_tlm(msid=m, epoch=_times[m][0])


    return processed_files


def _process_hdf(ingest_files, tstart, tstop, ingest_id, chunks, mdmap):

    global _times
    global _values

    # List of ingest files that have been processed
    # Later tar and move out of staging the files named in this list
    processed_files = []
    chunk_group = 0

    file_processing_queue = deque(ingest_files)
    queue_size = len(file_processing_queue)

    if sum(chunks) != queue_size:
        raise ValueError(f"sum of chunks ({sum(chunks)}) not equal processing queue size ({queue_size})")

    # Assume 60 files per ingest
    while len(file_processing_queue) != 0:
        _reset_storage()

        chunk = chunks[chunk_group]

        file_processing_chunk = [
            file_processing_queue.popleft()
            for i in range(chunk)
        ]
        
        chunk_group += chunk

        # for each group of files 1 - N where N <= 6
        for ingest_file in file_processing_chunk:
            try:
                with h5py.File(ingest_file['filename'], 'r') as f:
                    s = f['samples']
                    # get the msids that will be updated from this file
                    ids = pd.DataFrame(f['metadata'][...].byteswap().newbyteorder())['id'].to_list()
                    try:
                        # for each msid go through each dataset and append the data to write
                        for id in ids:
                            for dset in s.keys():
                                df = pd.DataFrame(s[dset][...].byteswap().newbyteorder())
                                tlm = df.loc[df['id']==id, ['observatoryTime', 'engineeringNumericValue']]
                                _times[mdmap[id]] = np.concatenate((_times[mdmap[id]], tlm['observatoryTime'].to_numpy()), 0)
                                _values[mdmap[id]] = np.concatenate((_values[mdmap[id]], tlm['engineeringNumericValue'].to_numpy()), 0)
                    except Exception as err:
                        raise err
            except (IOError, FileNotFoundError, Exception) as err:
                raise err
              
        for msid in _times.keys():
            with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
                npt = h5[msid].attrs['numpy_datatype'].replace('np.', '')
                _times[msid] = Time([t/1000 for t in _times[msid]], format='unix').jd
                sort_msid_data_by_time(msid)
                _values[msid] = _values[msid].astype(npt)
                _append_h5_col_tlm(msid=msid, epoch=_times[msid][0])

            processed_files.append(f)


def execute(ingest_type='CSV'):
    """ Perform one full update of the data archive based on parameters.

    This may be called in a loop by the program-level main().
    """
    logger.info('INGEST BEGIN >>>')
    logger.info('Ingest Module: {}'.format(os.path.abspath(__file__)))
    logger.info('Fetch Module: {}'.format(os.path.abspath(fetch.__file__)))

    # processed_msids = [x for x in pickle.load(open(msid_files['colnames'].abs, 'rb'))
    #             if x not in fetch.IGNORE_COLNAMES]

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        i = [h5[msid].attrs['id'] for msid in h5.keys()]
        mdmap = {id:name for id, name in zip(i, h5.keys())}

    _start_ingest_pipeline(mdmap=mdmap, ingest_type=ingest_type)

    logger.info(f'INGEST COMPLETE <<<')


if __name__ == "__main__":
    execute()
