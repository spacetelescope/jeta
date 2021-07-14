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
from Chandra.Time import DateTime

# import Ska.File
import Ska.DBI
import Ska.Numpy
import pyyaks.logger
import pyyaks.context

import h5py
import tables
import numpy as np

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


def _create_archive_database():
    """ Make empty files archive.meta.info.db3 if it doesn't exist
    """
    
    if not os.path.exists(msid_files['archfiles'].abs):
        archfiles_def = open(
                        get_env_variable('JETA_ARCHIVE_DEFINITION_SOURCE')
            ).read()
        filename = msid_files['archfiles'].abs
        logger.info('ALTERING ARCHIVE: Creating db {}'.format(filename))
        db = Ska.DBI.DBI(dbi='sqlite', server=filename, autocommit=False)
        db.execute(archfiles_def)
        db.commit()


def _create_msid_time_dataset(msid, h5):

    ft['msid'] = msid
    avg_msids_rows_per_ingest = _values[msid].size

    expectedrows = (avg_msids_rows_per_ingest
                    * FILES_IN_A_YEAR
                    * MISSION_LIFE_IN_YEARS)
    h5shape = (0,)
    h5type = tables.Atom.from_dtype(np.dtype('float64'))
    filters = tables.Filters(complevel=5, complib='zlib')

    h5.create_earray(
        h5.root,
        'time',
        h5type,
        h5shape,
        title=msid,
        expectedrows=expectedrows,
        filters=filters
    )

    return 0


def _create_msid_value_dataset(msid, h5):

    ft['msid'] = msid
    avg_msids_rows_per_ingest = _values[msid].size

    expectedrows = (avg_msids_rows_per_ingest
                    * FILES_IN_A_YEAR
                    * MISSION_LIFE_IN_YEARS)
    h5shape = (0,)

    # FIXME: The h5type fpr the msid values should vary
    # depending on the actual data to be stored.
    h5type = tables.Atom.from_dtype(np.dtype('float64'))
    filters = tables.Filters(complevel=5, complib='zlib')

    h5.create_earray(
        h5.root,
        'data',
        h5type,
        h5shape,
        title=msid,
        expectedrows=expectedrows,
        filters=filters
    )

    return 0


def _create_msid_directories(msids):
    """Create directories in the archive give a list of msids

        Parameters
        ----------
        msids : a python list of msids to add to the archive

    """

    for msid in msids:
        ft['msid'] = msid
        msid_directory_path = msid_files['msid'].abs
        if not os.path.exists(msid_directory_path):
            logger.info(f"ALTERING ARCHIVE: Creating new archive directory for {msid} ...")
            os.makedirs(msid_directory_path)


def _create_archive_files(msids):
    for msid in msids:
        ft['msid'] = msid

        if not os.path.exists(msid_files['mnemonic_value'].abs):
            logger.info((
                f"ALTERING ARCHIVE: Creating new archive files"
                f"(times.h5, values.h5) for {msid} ... "
            ))

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


def _aggregate_dataset_samples(samples, large_sample, offset):
    """Aggregates the dataset samples into a single ndarry from an HDF5 group

        Parameters
        ----------
        samples : <class 'h5py._hl.group.Group'>
                  `samples` contains a collection of 1 - n
                   datasets of msid samples
        large_sample : ndarray - dtype=[
                                            ('id', '>i8'),
                                            ('observatoryTime', '>i8'),
                                            ('groundTime', '>i8'),
                                            ('apid', '>i2'),
                                            ('engineeringNumericValue', '>f8'),
                                            ('engineeringTextValue', 'S80'),
                                            ('alarmStatus', '>i2')
                                        ]
                        `large_sample` is a datastructure used to collect all
                        tlm data from different files in memory.
        offset : int
                 This parameter is used to track where in `large_sample`
                 to start appending new data.

        Returns
        -------
        large_sample
            A pointer to the data being collected
        offset
            A updated `offset` values base on the number of rows
            appended thus far in the ingest.
    """

    grouped_sample = np.empty((0,), dtype=[
        ('id', '>i8'),
        ('observatoryTime', '>i8'),
        ('groundTime', '>i8'),
        ('apid', '>i2'),
        ('engineeringNumericValue', '>f8'),
        ('engineeringTextValue', 'S80'),
        ('alarmStatus', '>i2')]
    )

    for i in range(1, len(samples)+1):
        data = samples[f'data{i}']
        dset = data[...]
        grouped_sample = np.concatenate((grouped_sample, dset))

    for i, v in enumerate(grouped_sample):
        large_sample[i + offset] = v

    offset += len(grouped_sample)

    return large_sample, offset


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
        _create_msid_value_dataset(msid, values_h5)

    times_h5 = tables.open_file(
        f"{TELEMETRY_ARCHIVE}/data/tlm/{msid}/times.h5",
        mode='a'
    )

    if times_h5.__contains__('/times') is False:
        _create_msid_time_dataset(msid, times_h5)

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

    filename = os.path.basename(ingest_file_path)
    if db.fetchall('SELECT filename FROM archfiles WHERE filename=?', (filename,)):
        logger.verbose('File %s already in archfiles - unlinking and skipping' % filename)
        os.unlink(ingest_file_path)
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


def sort_msid_data_by_time(msid, times, values):

    global _times
    global _values

    _times[msid] = np.append(_times[msid], times)
    _values[msid] = np.append(_values[msid], values)

    msid_times = _times[msid]
    msid_values = _values[msid]

    idxs = torch.argsort(torch.from_numpy(msid_times))

    _times[msid] = np.array(msid_times[idxs])
    _values[msid] = np.array(msid_values[idxs])


def _allocate_large_sample(preallocation_size):

    return np.empty((preallocation_size,), dtype=[
        ('id', '>i8'),
        ('observatoryTime', '>i8'),
        ('groundTime', '>i8'),
        ('apid', '>i2'),
        ('engineeringNumericValue', '>f8'),
        ('engineeringTextValue', 'S80'),
        ('alarmStatus', '>i2')
    ])


def _organize_data_for_append(large_sample, mdmap):

    missing_ids = []

    buckets = {
        'times': defaultdict(list),
        'values': defaultdict(list)
    }

    for data in large_sample:
        if data['id'] == 0:
            continue
        try:
            msid = mdmap[data['id']]
            buckets['values'][msid].append(data['engineeringNumericValue'])
            # at this stage times are in milliseonds since unix epoch
            # i.e. 1970-01-01 00:00:00.000
            buckets['times'][msid].append(data['observatoryTime'])
            # _epochs[name] = _epochs[name] + len(_times[name])
        except Exception as err:
            logger.error(f"Error: {err}")
            missing_ids.append(data['id'])

    return buckets, missing_ids


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


def process_ingest_files(files_to_process, tstart, tstop, ingest_id, chunks):

    # List of ingest files that have been processed
    # Later tar and move out of staging the files named in this list
    processed_files = []
    chunk_group = 0

    db = Ska.DBI.DBI(
        dbi='sqlite',
        server=msid_files['archfiles'].abs,
        autocommit=False
    )

    ingest_record = {
        'processed_files': len(processed_files),
        'tstart': Time(Time.now(), format="datetime").unix,
        'rowstart': None,
        'ingest_status': 'processing',
        'new_msids': 0,
        'chunk_size': chunk_group,
        'ingest_id': ingest_id
    }

    sql = (
        "UPDATE ingest_history "
        "SET "
        f"processed_files={len(processed_files)}, "
        f"tstart={ingest_record['tstart']}, "
        f"rowstart='{ingest_record['rowstart']}', "
        f"ingest_status='{ingest_record['ingest_status']}', "
        f"new_msids={ingest_record['new_msids']}, "
        f"chunk_size={ingest_record['chunk_size']} "
        f"WHERE ingest_id='{ingest_record['ingest_id']}'"
    )
    db.execute(sql)
    db.commit()

    #
    file_processing_queue = deque(files_to_process)

    #
    original_queue_length = len(file_processing_queue)

    while len(file_processing_queue) != 0:

        mdmap = {}
        msids = []
        offset = 0

        db = Ska.DBI.DBI(
            dbi='sqlite',
            server=msid_files['archfiles'].abs,
            autocommit=False
        )

        out = db.fetchone('SELECT max(chunk_group) FROM archfiles')
        row = out['max(chunk_group)'] or 0

        last_archfile = db.fetchone('SELECT * FROM archfiles where chunk_group=?', (row,))

        # Get the list of msids that are already in the archive
        with open(msid_files['colnames'].abs, 'rb') as f:
            colnames = pickle.load(f)
            old_colnames = colnames.copy()

        _reset_storage()

        # if len(file_processing_queue) < chunk:
        #     chunk = len(file_processing_queue)

        chunk = chunks[files_to_process[chunk_group]['tstart']]

        file_processing_chunk = [
            file_processing_queue.popleft()
            for i in range(chunk)
        ]

        chunk_group += chunk

        # Sum the number of points for a chunk to get pre-allocation value
        num_points_in_chunk = sum([i['numPoints'] for i in file_processing_chunk])
        num_points_in_chunk = num_points_in_chunk + (num_points_in_chunk * .01)

        large_sample = _allocate_large_sample(
            int(num_points_in_chunk)
        )

        metadata = np.empty((0,), dtype=[
            ('name', 'S80'),
            ('id', '>i8'),
            ('hasEngNumeric', '>i2'),
            ('hasEngText', '>i2')
        ])

        logger.info(
            f"PROCESSING: chunk={len(file_processing_chunk)}, files processed="
            f"{len(processed_files)}/{original_queue_length}"
        )

        for ingest_file in file_processing_chunk:

            try:
                f = h5py.File(ingest_file['filename'], 'r')
            except (IOError, FileNotFoundError) as err:
                raise err

            try:
                large_sample, offset = _aggregate_dataset_samples(f['samples'], large_sample, offset)
                metadata = np.unique(np.concatenate((metadata, f['metadata'][...]), 0))
            except Exception as err:
                raise err


            yday = Time(ingest_file['tstart'], format='unix').yday
            archfiles_row = dict(
                filename=str(f.filename).replace('/srv/telemetry/staging/', ''),
                tstart=ingest_file['tstart'],
                tstop=ingest_file['tstop'],
                offset=offset,
                chunk_group=chunk_group,
                year=yday[0:4],
                doy=yday[5:8],
                processing_date=Time.now().iso,
                ingest_id=str(ingest_id)
            )
            db.insert(archfiles_row, 'archfiles')
            f.close()

        mdmap = {id:name.decode('ascii') for id, name in zip(metadata['id'], metadata['name'])}

        # a list of all the unique msids that a part of this update.
        # i.e. to be update with new data
        msids = list(mdmap.values())

        # a list of msids that are new to the archive
        new_msids = np.setdiff1d(msids, colnames)
        # TODO: Replace with num_points_in_chunk = int(1.01*num_points_in_chunk)
        ingest_record['new_msids'] = int(ingest_record['new_msids']) + len(new_msids)

        # update this list of msids stored in the archive
        if new_msids.tolist():
            colnames.update(new_msids)
            # Create any msid archive directories that do not already exists
            _create_msid_directories(new_msids)
            # Create any msid archive files that do not already exists
            _create_archive_files(new_msids)


        # If colnames changed then give warning and update files.
        if colnames != old_colnames:
            logger.warning(f"WARNING: updating {msid_files['colnames'].abs} because mnemonic names changed ...")
            with open(msid_files['colnames'].abs, 'wb') as f:
                pickle.dump(colnames, f)

        logger.info(
            f"PREPROCESSING: Organizing data into buckets to append {len(large_sample)}"
            f" new datapoints to the archive for {len(msids)} msids ..."
        )

        bucket, missing_ids = _organize_data_for_append(
            large_sample=large_sample,
            mdmap=mdmap
        )

        if len(missing_ids) != 0:
            logger.warning(f"!!! MISSING MSIDS: {len(missing_ids)} msids from the map!!!")

        logger.info(
            f"UPDATING: Sorting then appending from sample size={len(large_sample)}"
            f" msids={len(msids)}"
        )

        # more or less create function 'pointers' for speed (i.e. python doesn't have to look up the address every iteration)
        append_func = _append_h5_col_tlm
        sort_func = sort_msid_data_by_time
        delta_funk = calculate_delta_times

        for msid in msids:
            try:
                sort_func(msid, bucket['times'][msid], bucket['values'][msid])
                # if _times[msid].ndim == 0:
                #     logger.warning(f'{msid} times data has diminsion {_times[msid].ndim}')
                #     t0 = _times[msid]
                #     epoch = Time([t0/1000.0], format='unix').jd
                # else:
                t0 = np.atleast_1d(_times[msid])[0]
                epoch = Time(t0/1000.0, format='unix').jd
                delta_funk(msid, _times[msid], epoch)

                # Data preprocessing has ended, its now time to write to the archive.
                append_func(msid, epoch)

            except Exception as err:
                # TODO: Write tests for edge cases and develop exception handling scheme.
                # maybe write ingest_failed() function as well to encapsulate clean up ans roll back.
                logger.error(f"!!! CRITICAL ERROR !!! failed to sort and calculate times for {msid} | {_times[msid].ndim} | {_times[msid]} | {type(_times[msid])}")
                raise err

        processed_files = processed_files + file_processing_chunk

        sql = (
            "UPDATE ingest_history "
            "SET "
            f"processed_files={len(processed_files)} "
            f"WHERE ingest_id='{ingest_record['ingest_id']}'"
        )
        db.execute(sql)
        db.commit()

    ingest_record['tstop'] = Time(Time.now(), format="datetime").unix
    ingest_record['processed_files'] = len(processed_files)
    ingest_record['ingest_status'] = 'success'

    db = Ska.DBI.DBI(
        dbi='sqlite',
        server=msid_files['archfiles'].abs,
        autocommit=False
    )
    sql = (
        "UPDATE ingest_history "
        "SET "
        f"processed_files={ingest_record['processed_files']}, "
        f"tstop={ingest_record['tstop']}, "
        f"ingest_status='{ingest_record['ingest_status']}', "
        f"new_msids={ingest_record['new_msids']} "
        f"WHERE ingest_id='{ingest_record['ingest_id']}'"
    )
    db.execute(sql)
    db.commit()

    return processed_files


def move_archive_files(filetype, processed_ingest_files):

    if processed_ingest_files is not None:

        import tarfile
        import shutil

        os.chdir(STAGING_DIRECTORY)

        tarfile_name = f"stage_{int(DateTime().secs)}.tar"
        tar = tarfile.open(tarfile_name, mode='w')

        for ingest_file in processed_ingest_files:
            tar.add(ingest_file)
            os.remove(ingest_file)

        tar.close()
        shutil.move(
            tarfile_name,
            f"{msid_files['processed_files_directory'].abs}/{tarfile_name}"
        )


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

        # Disable auto-commit to enable rollback
        db = Ska.DBI.DBI(
            dbi='sqlite',
            server=msid_files['archfiles'].abs,
            autocommit=False
        )
        # Record the start of the ingest in the SQLite DB
        ingest_record = {
            'discovered_files': len(ingest_files),
            'tstart': -1,
            'tstop': -1,
            'coverage_start': -1,
            'coverage_end': -1,
            'ingest_status': 'starting',
            'ingest_id': str(ingest_id.int),
            'uuid': str(ingest_id)
        } 
        db.insert(ingest_record, 'ingest_history')
        db.commit()

        if ingest_type == 'csv' or ingest_type == 'CSV':
            import pandas as pd

            for f in ingest_files:
                fof = pd.read_csv(f) 
                from astropy.time import Time
                
                _reset_storage()
                msid = fof['Telemetry Mnemonic'][0]
                print('here 1')
                with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
                    npt = h5[msid].attrs['numpy_datatype'].replace('np.', '')
                    print('here 2')
                    
                    times = fof['Observatory Time'].str.replace('/', '-')
                    times = Time(times.to_list() , format='iso').jd
                    
                    print('here 3')
                    sort_msid_data_by_time(msid, times, fof['EU Value'].to_list())
                    _values[msid] = _values[msid].astype(npt)
                    # _times[msid] = np.diff(np.insert(_times[msid], 0, _times[msid][0])) 
                    print('here 4')
                    _append_h5_col_tlm(msid=msid, epoch=_times[msid][0])
                    print('here 5')
            
        elif ingest_type == 'h5':
            ingest_files = _sort_ingest_files_by_start_time(ingest_files)
            # HDF5 files have been given with multiple identical start times
            # with different end times. Those files are grouped together here.
            chunks = _get_ingest_chunk_sequence(ingest_files)
            out = db.fetchone('SELECT count(*) FROM ingest_history')
            processed_ingest_files = _process_hdf5(
                ingest_files,
                ingest_files[0]['tstart'],
                ingest_files[-1]['tstop'],
                ingest_id=ingest_id.int,
                chunks=chunks
            ) 
        else:
            raise ValueError('Ingest type parameter is invalid neither csv or h5 was used.')
            
        # move_archive_files(filetype, processed_ingest_files)
    else:
        logger.info('No ingest files discovered in {}')

def _process_csv():
    pass

def _process_hdf5(files_to_process, tstart, tstop, ingest_id, chunks):

    # List of ingest files that have been processed
    # Later tar and move out of staging the files named in this list
    processed_files = []
    chunk_group = 0

    db = Ska.DBI.DBI(
        dbi='sqlite',
        server=msid_files['archfiles'].abs,
        autocommit=False
    )

    ingest_record = {
        'processed_files': len(processed_files),
        'tstart': Time(Time.now(), format="datetime").unix,
        'rowstart': None,
        'ingest_status': 'processing',
        'new_msids': 0,
        'chunk_size': chunk_group,
        'ingest_id': ingest_id
    }

    sql = (
        "UPDATE ingest_history "
        "SET "
        f"processed_files={len(processed_files)}, "
        f"tstart={ingest_record['tstart']}, "
        f"rowstart='{ingest_record['rowstart']}', "
        f"ingest_status='{ingest_record['ingest_status']}', "
        f"new_msids={ingest_record['new_msids']}, "
        f"chunk_size={ingest_record['chunk_size']} "
        f"WHERE ingest_id='{ingest_record['ingest_id']}'"
    )
    db.execute(sql)
    db.commit()

    #
    file_processing_queue = deque(files_to_process)

    #
    original_queue_length = len(file_processing_queue)

    while len(file_processing_queue) != 0:

        mdmap = {}
        msids = []
        offset = 0

        db = Ska.DBI.DBI(
            dbi='sqlite',
            server=msid_files['archfiles'].abs,
            autocommit=False
        )

        out = db.fetchone('SELECT max(chunk_group) FROM archfiles')
        row = out['max(chunk_group)'] or 0

        last_archfile = db.fetchone('SELECT * FROM archfiles where chunk_group=?', (row,))

        # Get the list of msids that are already in the archive
        with open(msid_files['colnames'].abs, 'rb') as f:
            colnames = pickle.load(f)
            old_colnames = colnames.copy()

        _reset_storage()

        # if len(file_processing_queue) < chunk:
        #     chunk = len(file_processing_queue)

        chunk = chunks[files_to_process[chunk_group]['tstart']]

        file_processing_chunk = [
            file_processing_queue.popleft()
            for i in range(chunk)
        ]

        chunk_group += chunk

        # Sum the number of points for a chunk to get pre-allocation value
        num_points_in_chunk = sum([i['numPoints'] for i in file_processing_chunk])
        num_points_in_chunk = num_points_in_chunk + (num_points_in_chunk * .01)

        large_sample = _allocate_large_sample(
            int(num_points_in_chunk)
        )

        metadata = np.empty((0,), dtype=[
            ('name', 'S80'),
            ('id', '>i8'),
            ('hasEngNumeric', '>i2'),
            ('hasEngText', '>i2')
        ])

        logger.info(
            f"PROCESSING: chunk={len(file_processing_chunk)}, files processed="
            f"{len(processed_files)}/{original_queue_length}"
        )

        for ingest_file in file_processing_chunk:

            try:
                f = h5py.File(ingest_file['filename'], 'r')
            except (IOError, FileNotFoundError) as err:
                raise err

            try:
                large_sample, offset = _aggregate_dataset_samples(f['samples'], large_sample, offset)
                metadata = np.unique(np.concatenate((metadata, f['metadata'][...]), 0))
            except Exception as err:
                raise err


            yday = Time(ingest_file['tstart'], format='unix').yday
            archfiles_row = dict(
                filename=str(f.filename).replace('/srv/telemetry/staging/', ''),
                tstart=ingest_file['tstart'],
                tstop=ingest_file['tstop'],
                offset=offset,
                chunk_group=chunk_group,
                year=yday[0:4],
                doy=yday[5:8],
                processing_date=Time.now().iso,
                ingest_id=str(ingest_id)
            )
            db.insert(archfiles_row, 'archfiles')
            f.close()

        mdmap = {id:name.decode('ascii') for id, name in zip(metadata['id'], metadata['name'])}

        # a list of all the unique msids that a part of this update.
        # i.e. to be update with new data
        msids = list(mdmap.values())

        # a list of msids that are new to the archive
        new_msids = np.setdiff1d(msids, colnames)
        # TODO: Replace with num_points_in_chunk = int(1.01*num_points_in_chunk)
        ingest_record['new_msids'] = int(ingest_record['new_msids']) + len(new_msids)

        # update this list of msids stored in the archive
        if new_msids.tolist():
            colnames.update(new_msids)
            # Create any msid archive directories that do not already exists
            _create_msid_directories(new_msids)
            # Create any msid archive files that do not already exists
            _create_archive_files(new_msids)


        # If colnames changed then give warning and update files.
        if colnames != old_colnames:
            logger.warning(f"WARNING: updating {msid_files['colnames'].abs} because mnemonic names changed ...")
            with open(msid_files['colnames'].abs, 'wb') as f:
                pickle.dump(colnames, f)

        logger.info(
            f"PREPROCESSING: Organizing data into buckets to append {len(large_sample)}"
            f" new datapoints to the archive for {len(msids)} msids ..."
        )

        bucket, missing_ids = _organize_data_for_append(
            large_sample=large_sample,
            mdmap=mdmap
        )

        if len(missing_ids) != 0:
            logger.warning(f"!!! MISSING MSIDS: {len(missing_ids)} msids from the map!!!")

        logger.info(
            f"UPDATING: Sorting then appending from sample size={len(large_sample)}"
            f" msids={len(msids)}"
        )

        # more or less create function 'pointers' for speed (i.e. python doesn't have to look up the address every iteration)
        append_func = _append_h5_col_tlm
        sort_func = sort_msid_data_by_time
        delta_funk = calculate_delta_times

        for msid in msids:
            try:
                sort_func(msid, bucket['times'][msid], bucket['values'][msid])
                # if _times[msid].ndim == 0:
                #     logger.warning(f'{msid} times data has diminsion {_times[msid].ndim}')
                #     t0 = _times[msid]
                #     epoch = Time([t0/1000.0], format='unix').jd
                # else:
                t0 = np.atleast_1d(_times[msid])[0]
                epoch = Time(t0/1000.0, format='unix').jd
                delta_funk(msid, _times[msid], epoch)

                # Data preprocessing has ended, its now time to write to the archive.
                append_func(msid, epoch)

            except Exception as err:
                # TODO: Write tests for edge cases and develop exception handling scheme.
                # maybe write ingest_failed() function as well to encapsulate clean up ans roll back.
                logger.error(f"!!! CRITICAL ERROR !!! failed to sort and calculate times for {msid} | {_times[msid].ndim} | {_times[msid]} | {type(_times[msid])}")
                raise err

        processed_files = processed_files + file_processing_chunk

        sql = (
            "UPDATE ingest_history "
            "SET "
            f"processed_files={len(processed_files)} "
            f"WHERE ingest_id='{ingest_record['ingest_id']}'"
        )
        db.execute(sql)
        db.commit()

    ingest_record['tstop'] = Time(Time.now(), format="datetime").unix
    ingest_record['processed_files'] = len(processed_files)
    ingest_record['ingest_status'] = 'success'

    db = Ska.DBI.DBI(
        dbi='sqlite',
        server=msid_files['archfiles'].abs,
        autocommit=False
    )
    sql = (
        "UPDATE ingest_history "
        "SET "
        f"processed_files={ingest_record['processed_files']}, "
        f"tstop={ingest_record['tstop']}, "
        f"ingest_status='{ingest_record['ingest_status']}', "
        f"new_msids={ingest_record['new_msids']} "
        f"WHERE ingest_id='{ingest_record['ingest_id']}'"
    )
    db.execute(sql)
    db.commit()

    return processed_files

def execute():
    """ Perform one full update of the data archive based on parameters.

    This may be called in a loop by the program-level main().
    """
    logger.info('INGEST BEGIN >>>')
    logger.info('Ingest Module: {}'.format(os.path.abspath(__file__)))
    logger.info('Fetch Module: {}'.format(os.path.abspath(fetch.__file__)))

    filetypes = fetch.filetypes
    # processed_msids = [x for x in pickle.load(open(msid_files['colnames'].abs, 'rb'))
    #             if x not in fetch.IGNORE_COLNAMES]

    _create_archive_database()

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        i = [h5[msid].attrs['id'] for msid in h5.keys()]
        mdmap = {id:name for id, name in zip(i, h5.keys())}

    _start_ingest_pipeline(mdmap=mdmap, ingest_type='CSV')

    logger.info(f'INGEST COMPLETE <<<')

if __name__ == "__main__":
    execute()
