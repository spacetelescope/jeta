#!/usr/bin/env python
# see LICENSE.rst

from __future__ import print_function, division, absolute_import
from io import FileIO

import os
import sys
import getopt
import glob
from uuid import uuid1
from functools import partial
import pickle
from random import seed
import datetime
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
from torch._C import Value

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

BAD_APID_LIST = [712]


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


def _append_h5_col_tlm(msid, epoch, times=None, values=None, apply_direct=False):

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
    # TODO: Filter interval before append
    # if index != 0:
    #     last_time = times_h5.root.times[index -  1]
    #     tstart = np.atleast_1d(_times[msid])[0]
    #     tstop = np.atleast_1d(_times[msid])[-1]

    try:
        if apply_direct == True:
            values_h5.root.values.append(np.atleast_1d(values))
            times_h5.root.times.append(np.atleast_1d(times)) 
        else:
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


def sort_msid_data_by_time(mid, times=None, values=None, append=True):

    global _times
    global _values

    if append:
        pass
        # FIXME: this may not be require with the new design
        # _times[mid] = np.append(_times[mid], times)
        # _values[mid] = np.append(_values[mid], values)

    idxs = torch.argsort(torch.from_numpy(_times[mid]))

    _times[mid] = _times[mid][idxs]
    _values[mid] = _values[mid][idxs]

def _sort_ingest_files_by_start_time(list_of_files=[], data_origin='OBSERVATORY'):
    
    # retrieve environment variables
    BYPASS_GAP_CHECK = int(os.environ.get('JETA_BYPASS_GAP_CHECK', False))
    BYPASS_DURATION_CHECK = int(os.environ.get('JETA_BYPASS_DURATION_CHECK', False))
    
    # TODO: Move epoch to system config
    epoch = datetime.datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    ingest_list = []

    for file in list_of_files:
        with h5py.File(file, 'r') as f:
            # if the files data origin is not correct move on 
            # to the next one.
            if data_origin not in str(f.attrs['/dataOrigin'][0]):
                logger.info("{} skipped due to incorrect data origin {}".format(file, str(f.attrs['/dataOrigin'][0]))) # LITA-181
                continue

            df = None
            for dataset in f['samples'].keys():
                dff = pd.DataFrame( np.array(f['samples'][dataset]).byteswap().newbyteorder() )            
                dff = dff.loc[(dff['id'] != 0) & (dff['apid'] > 0) & (~dff['apid'].isin(BAD_APID_LIST))]
                df = pd.concat([df, dff])

            # don't consider data before the mission epoch
            df = df.loc[(df['observatoryTime'] > 1640304000000)]  
            tstart = df['observatoryTime'].min()/1000
            tstop = df['observatoryTime'].max()/1000
            
            try:
                dt_tstart = epoch + datetime.timedelta(seconds=int(tstart))
                dt_tstop = epoch + datetime.timedelta(seconds=int(tstop))
                
                # perform time check, LITA-213
                if BYPASS_DURATION_CHECK or ((dt_tstop < datetime.datetime.now()) and ((dt_tstop - dt_tstart) < datetime.timedelta(hours=4))):
                    ingest_list.append(
                        {
                            'filename': f.filename,
                            'tstart': tstart,
                            'tstop': tstop,
                            'numPoints': f.attrs['/numPoints']
                        }
                    )
                else:
                    logger.info( f"{file}: time check violated. Duration {dt_tstop - dt_tstart}, stop time {dt_tstop}" )
                    
            except Exception as e:
                logger.info("{}, {}".format(file, e))
                

    if not ingest_list:
        # cancel ingest, LITA-181
        return []
    
    #sort by start time, and secondarily by stop time. 
    ingest_list = sorted(ingest_list, key=lambda k: k['tstop'] )    
    
    if len(ingest_list) > 120:
        # LITA-182
        ingest_list = ingest_list[:120]

    # perform gap checking per LITA-179
    if BYPASS_GAP_CHECK:
        logger.info("Gap check is bypassed. No check performed.")
    
    else:
        ingest_list = _ingest_list_gap_check(ingest_list)
    
    if not ingest_list:
        return []
    
    dt_tstart = epoch + datetime.timedelta(seconds=int(ingest_list[0]['tstart']))
    dt_tstop = epoch + datetime.timedelta(seconds=int(ingest_list[-1]['tstop']))

    logger.info("=-=-=-=-=-=-=-=-=-=-=-=INGEST FILE(S) COVERAGE REPORT-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    for f in ingest_list: 
        logger.info(("{}, {}, {}").format(
            os.path.basename(f['filename']), 
            (epoch + datetime.timedelta(seconds=int(f['tstart']))).strftime('%Y:%j:%H:%M:%S'), 
            (epoch + datetime.timedelta(seconds=int(f['tstop']))).strftime('%Y:%j:%H:%M:%S'))
        )
    logger.info(
        (
            "Full Data Coverage for all files (tstart, tstop): "
            "({},"
            "{})"
        ).format(dt_tstart.strftime('%Y:%j:%H:%M:%S'), dt_tstop.strftime('%Y:%j:%H:%M:%S'))
    )

    return ingest_list


def _ingest_list_gap_check(ingest_list):

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'a') as ref_data:
        
        if not 'last_ingested_timestamp' in list(ref_data.attrs):
            # initialize attribute if not in archive
            ref_data.attrs['last_ingested_timestamp'] = Time(ingest_list[0]['tstart'], format='unix').jd

        last_ingested_timestamp = Time(ref_data.attrs['last_ingested_timestamp'], format='jd').unix   
        if ( ingest_list[0]['tstart'] - last_ingested_timestamp ) > 1:
            logger.info("Cancelling ingest. Gap since previous ingest.")
            logger.info("Last ingested timestamp: {}".format(Time(last_ingested_timestamp, format='unix').yday) )
            logger.info( "Fist timestamp in {}: {}".format( ingest_list[0]['filename'], Time(ingest_list[0]['tstart'], format='unix').yday ))
            return []
        
        for i in range(1, len(ingest_list)):
            if (ingest_list[i]['tstart'] - ingest_list[i-1]['tstop']) > 1:
                logger.info("Truncating ingest list due to gap before {}".format(ingest_list[i]['filename']) ) 
                ingest_list = ingest_list[:i]
                break
                
    return ingest_list


def _auto_file_discovery(ingest_type, source_type):
    """ Automatically discover and return a list of files to ingest in the staging area.

    Parameters
    ==========
        ingest_type : str
            the type of file to match in the file search. 
            examples of valid values are: h5, hdf, csv, txt
        source_type : str
            the source type of the data e.g. (FILE_PREFIX, Directory Path or Sim vs. Flight)
            source_type of an empty string means sources of ingest_type in default staging.

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


def move_archive_files(processed_files):

    if processed_files is not None:

        import tarfile
        import shutil

        os.chdir(os.environ['STAGING_DIRECTORY'])

        tarfile_name = f"processed_files_{Time(Time.now()).yday.replace(':','').replace('.','')}.tar.gz"
        
        tar = tarfile.open(tarfile_name, mode='x:gz')
        for ingest_file in processed_files:
            tar.add(ingest_file['filename'])
            os.remove(ingest_file['filename'])
        tar.close()

        shutil.copyfile(
            f"{os.environ['STAGING_DIRECTORY']}{tarfile_name}",
            f"{os.environ['TELEMETRY_ARCHIVE']}/{tarfile_name}"
        )
        os.remove(tarfile_name)


def _ingest_virtual_dataset(ref_data, mdmap, vds_tstart=None, vds_tstop=None):
    """ Do the actual work of extracting data from the files
        and appending it to the archive.

        Parameters:
            ref_data: 
                a data structure containing reference meta data
                about each msid.
            mdmap:
                a dict with ids as keys and msid friendly strings 
                as keys
    """
    logger.info('Ingesting virtual dataset temp file VDS.hdf5')
    with h5py.File("VDS.hdf5", 'r', libver='latest') as vds:
        # Get all the MSIDs data from the VDS (Virtual Data Set) 
        # and use it to create a Pandas Dataframe
        df = pd.DataFrame(vds['/data'][...].byteswap().newbyteorder())
        
        # Entries with ID 0 are padding and can be ignored
        ids = df['id'].unique()
        ids = ids[ids != 0]
        ids = np.intersect1d(ids, np.array(list(mdmap.keys()),dtype=int))

        # Remove samples with id == 0
        df = df.loc[(df['id'] != 0)]

        # Remove duplicate entries
        df.drop_duplicates(subset=['id', 'observatoryTime', 'engineeringNumericValue'], inplace=True)

        df = df.sort_values(by=['observatoryTime'])
        df['observatoryTime'] = Time(df['observatoryTime']/1000, format='unix').jd
        
        if vds_tstart and vds_tstop:
            vds_tstart = Time(vds_tstart, format='unix').jd
            vds_tstop = Time(vds_tstop, format='unix').jd
            df = df.loc[(df['observatoryTime'] >= vds_tstart) & (df['observatoryTime'] <= vds_tstop)]
        
        df = df.groupby(["id"])[['observatoryTime', 'engineeringNumericValue', 'apid']]
        
        for msid_id in ids:
            tlm = df.get_group(msid_id)

            if 'last_ingested_timestamp' not in list(ref_data[mdmap[msid_id]].attrs):
                # Set the default value to DEC 24 2021, 00:00:00
                ref_data[mdmap[msid_id]].attrs['last_ingested_timestamp'] = 2459572.5

            if tlm['observatoryTime'].min() <= ref_data[mdmap[msid_id]].attrs['last_ingested_timestamp']:
                if tlm['observatoryTime'].max() > ref_data[mdmap[msid_id]].attrs['last_ingested_timestamp']:
                    # remove overlap 
                    tlm = tlm.loc[tlm['observatoryTime'] > ref_data[mdmap[msid_id]].attrs['last_ingested_timestamp']]
                else:
                    # do not ingest. time range already covered.
                    continue
                
            times = tlm['observatoryTime'].to_numpy()
            values = tlm['engineeringNumericValue'].to_numpy()
            last_ingested_timestamp = times[-1]

            # Get this MSID numpy datatype
            # npt = ref_data[mdmap[msid_id]].attrs['numpy_datatype'].replace('np.', '')

            # Set datatypes for archive data
            times.dtype = np.float64
            # values = values.astype(npt)

            # Ensure data is sorted in time order
            # TODO: Verify this can be removed permanently
            # since data should come order by msid 
            # sort_msid_data_by_time(mid, append=False)

            # Finally append the data to the archive.
            _append_h5_col_tlm(msid=mdmap[msid_id], epoch=times[0], times=times, values=values, apply_direct=True)
            
            # update the saved last_ingested_timestamp after successful ingest (LITA-184)
            ref_data[mdmap[msid_id]].attrs['last_ingested_timestamp'] = last_ingested_timestamp # last timestamp in the ingest


def _preprocess_hdf(ingest_files):
    ingest_files = _sort_ingest_files_by_start_time(ingest_files)

    if not ingest_files:
        return [], []
    
    # HDF5 files have been given with multiple identical start times
    # with different end times. Those files are grouped together here. 
    chunks = _get_ingest_chunk_sequence(ingest_files)

    return ingest_files, chunks


def _start_ingest_pipeline(ingest_type="h5", source_type='E', provided_ingest_files=None, mdmap=None):
    """ This is the function internal to the system that archives data.

    Parameters
    ==========
    ingest_type : str
            this parameter impacts whichs filetype is gathered in a list to be ingested by the 
            automatic file discovery routine (i.e. jeta.archive.ingest.) .
    provided_ingest_files : list or list-like array of ingest files as string filenames.
            this is an optional parameter to supply a list of specific files to ingest.

    """
    
    f = lambda p: p if p is None else len(p)
    logger.info(f"Ingest Parameters: ingest_type -> {ingest_type}, source_type -> {source_type}, provided_ingest_files-> {f(provided_ingest_files)}")
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
            ingest_file_data, chunks = _preprocess_hdf(ingest_files)
            logger.info('Completed HDF5 file pre-processing ...')

            if not ingest_file_data:
                logger.info('Empty ingest list.') # LITA-181
                return

            logger.info('Starting HDF5 file data ingest ...')
            # out = db.fetchone('SELECT count(*) FROM ingest_history')
            processed_files = _process_hdf(
                ingest_file_data=ingest_file_data,
                mdmap=mdmap
            )
            logger.info('Completed HDF5 file data ingest ALL data sequence ...') 
        else:
            raise ValueError('Ingest type parameter is invalid. Valid options are csv or h5.')
        logger.info(f'Moving {len(processed_files)} HDF5 ingest file(s) to tmp storage ... ')
        move_archive_files(processed_files)
        
        
        if int(os.environ.get('JETA_UPDATE_STATS', True)):
            # Once data ingest is complete update the 5min and daily stats data
            from jeta.archive import update
            update.main()
        else:
            logger.info(f'Skipping stats update.') # LITA-191
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
                npt = np.float64, # h5[msid].attrs['numpy_datatype'].replace('np.', '')
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
                    npt = np.float64, # h5[m].attrs['numpy_datatype'].replace('np.', '')
                    times = fof.loc[fof['Telemetry Mnemonic']==m, ['Observatory Time']]
                    times = times['Observatory Time'].str.replace('/', '-')
                    times = Time(times.tolist() , format='iso').jd
                    values = fof.loc[fof['Telemetry Mnemonic']==m, ['EU Value']]
                    sort_msid_data_by_time(m, times, values.values.tolist())
                    _values[m] = _values[m].astype(npt)
                    _append_h5_col_tlm(msid=m, epoch=_times[m][0])

    return processed_files


def _process_hdf(ingest_file_data, mdmap):
    """ Function for handling the ingest of HDF5 files deliveried to the staging area.
    """
    
    MAX_FILE_PROCESSING_CHUNK = np.max(int(get_env_variable('JETA_INGEST_CHUNK_SIZE')), 20)

    # global _times
    # global _values

    processed_files = []
    total_data_points = None

    file_processing_queue = deque(ingest_file_data)
    queue_size = len(file_processing_queue)

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'a') as ref_data:
        chunk_seams = []
        files_processed_chunk = []
        while len(file_processing_queue) != 0:
            # Get a chunk of files to process, either the next 5 or whatever is left
            chunk = min(MAX_FILE_PROCESSING_CHUNK, len(file_processing_queue))
            file_processing_chunk = [
                file_processing_queue.popleft()
                for i in range(chunk)
            ]

            # clear array data before processing the next chunk
            # _reset_storage()

            dtype = None
            n_samples_to_ingest = 0

            for file_data in file_processing_chunk:
                with h5py.File(file_data['filename'], 'r') as current_ingest_file:
                    # FIXME: create config for dtype parameter, value rarely changes
                    dtype = current_ingest_file['samples']['data1'].dtype 
                    n_samples_to_ingest += sum([d.shape[0] for d in current_ingest_file['samples'].values()])

            # Create a layout to hold a memory map of all files in the chunk
            shape = (n_samples_to_ingest, )
            layout = h5py.VirtualLayout(shape=shape,
                                dtype=dtype)

            # For each file in the chunk, create a memory map that presents
            # the individual datasets in the file as one large dataset
            # then append the msid data to global arrays after checking for overlaps
            # and disgarding APID>0
            idx0 = 0
            for file_data in file_processing_chunk:
                try:
                    with h5py.File(file_data['filename'], 'r') as current_ingest_file:   
                        # Map datasets in the current file to add to the layout
                        for i, dset in enumerate(current_ingest_file['samples']):
                            vsource = h5py.VirtualSource(current_ingest_file['samples'][dset]) 
                            n_samples = vsource.shape[0]
                            # print(vsource.shape)
                            # print(n_samples)
                            layout[idx0:idx0+n_samples] = vsource
                            idx0 += n_samples
                        chunk_seams.append(idx0) # index of the end of the file?
                        files_processed_chunk.append(file_data['filename'])
                except Exception as err:
                    raise err

            # Filenames being processed in this chunk
            # print(files_processed_chunk)

            # Total numer of points eing processed in this chunk
            # print(layout.shape)

            # File seem indicies?
            # print(chunk_seams)
            # Create Virtual Dataset
            with h5py.File("VDS.hdf5", 'w', libver='latest') as vds:
                vds.create_virtual_dataset('/data', layout)
                logger.info("Created VDS containing: ")
                logger.info([os.path.basename(f['filename']) for f in file_processing_chunk])
            
            _ingest_virtual_dataset(ref_data, mdmap, vds_tstart=min([f['tstart'] for f in file_processing_chunk]), vds_tstop=max([f['tstop'] for f in file_processing_chunk]))
            
            # print(f"{layout.shape[0]} n_samples.")
            
            # store last timestamp from chunnk to ref_data, for gap checking
            # only update if last timestamp in chunck is greater than stored timestamp
            if Time(file_processing_chunk[-1]['tstop'], format='unix').jd > ref_data.attrs['last_ingested_timestamp']:
                ref_data.attrs['last_ingested_timestamp'] = Time(file_processing_chunk[-1]['tstop'], format='unix').jd

            processed_files.extend(file_processing_chunk)

    return processed_files
        

def execute(ingest_type='h5', source_type='E', provided_ingest_files=None):
    """ Perform one full update of the data archive based on parameters.

    This may be called in a loop by the program-level main().
    """
    logger.info('INGEST BEGIN >>>')
    logger.info('Ingest Module: {}'.format(os.path.abspath(__file__)))
    logger.info('Fetch Module: {}'.format(os.path.abspath(fetch.__file__)))
    logger.info('Loading system configuration ... ')

    from jeta.archive.operations import load_config
    current_settings = load_config()
    logger.info(f'System configuration loaded: \n{current_settings}')

    # processed_msids = [x for x in pickle.load(open(msid_files['colnames'].abs, 'rb'))
    #             if x not in fetch.IGNORE_COLNAMES]

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'r') as h5:
        i = [h5[msid].attrs['id'] for msid in h5.keys()]
        mdmap = {id:name for id, name in zip(i, h5.keys())}

    _start_ingest_pipeline(
        mdmap=mdmap, # map between msids names as strings and their numerical id
        ingest_type=ingest_type, # hdf5 or fof
        provided_ingest_files=provided_ingest_files, # a list of ingest files or None
        source_type=source_type # C, E, or R
    )

    logger.info(f'INGEST COMPLETE <<<')


if __name__ == "__main__":
    staging = os.environ['STAGING_DIRECTORY']
    provided_ingest_files=None

    if len(sys.argv) > 1:
        opts, args = getopt.getopt(sys.argv, 'i', 'ingest_files=')
        try:
           if '-i' in args:
                input_files = [file.replace('\n', '') for file in open(args[args.index('-i') + 1])]
                provided_ingest_files = [f for f in filter(lambda s: s != '' and s[0] != '#', input_files)]
                provided_ingest_files = ['{}{}{}'.format(staging,'/', f) for f in provided_ingest_files]
        except IOError as err:
            print(err.args[0])
    execute(ingest_type='h5', source_type='E', provided_ingest_files=provided_ingest_files)
