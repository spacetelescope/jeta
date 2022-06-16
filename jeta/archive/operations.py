import os
import json
import pickle
import sqlite3

import numpy as np

import h5py
import tables

from collections import defaultdict

# import pyyaks.logger
# import pyyaks.context

from astropy.time import Time

from jeta.archive.utils import get_env_variable

ENG_ARCHIVE = get_env_variable('ENG_ARCHIVE')
TELEMETRY_ARCHIVE = get_env_variable('TELEMETRY_ARCHIVE')
ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')

DEFAULT_CONFIG_PATH = '/srv/jeta/jeta/config/parameters.hdf5'


def get_default_config(config_path=DEFAULT_CONFIG_PATH) -> str:
    """ Get the default current config as json
        Parameters
        ----------
        :return: serialized json str of the current system configuration
    """
    default_settings = defaultdict(list)

    with h5py.File(config_path, 'r') as c:
        for k in c.keys():
            for a in c[k].attrs:
                default_settings[k].append({a.upper(): str(c[k].attrs[a])})
                os.environ[a.upper()] = str(c[k].attrs[a])
    return json.dumps(default_settings)


def set_config_parameter(
    subsystem: str = None,
    param: str = None,
    value: str = None
) -> None:
    """ Set the subsystem-level configuration in a persisted config file
        Parameters
        ----------
        :param subsystem: the target subsystem of the configuration parameter
        :param param: the name of the parameter value to set
        :param value: the value to set the parameter to

        :returns: config as str
    """
    if None in [subsystem, param, value]:
        raise ValueError('Subsystem, param, and value are required.')
    with h5py.File(f'{TELEMETRY_ARCHIVE}/config/parameters.hdf5', 'a') as c:
        if param.lower() in list(c[subsystem].attrs):
            c[subsystem].attrs[param.lower()] = value
        else:
            raise ValueError(f'{param} is not a valid system parameter.')


def get_config_parameter(subsystem: str = None, param: str = None) -> str:
    """ Get the subsystem-level configuration from persisted config file

        Parameters
        ----------
        :param subsystem: the target subsystem of the configuration parameter
        :param param: the name of the parameter value sought

        :returns: config as str
    """
    if None in [subsystem, param]:
        raise ValueError('Both subsystem and param are required.')
    with h5py.File(f'{TELEMETRY_ARCHIVE}/config/parameters.hdf5', 'r') as c:
        return c[subsystem].attrs[param.lower()]


def load_config() -> str:
    """ Load the current config into memory by setting as environment variables
        Parameters
        ----------
        :return: serialized json str of the current system configuration
    """
    current_settings = defaultdict(list)

    with h5py.File(f'{TELEMETRY_ARCHIVE}/config/parameters.hdf5', 'r') as c:
        for k in c.keys():
            for a in c[k].attrs:
                current_settings[k].append({a.upper(): str(c[k].attrs[a])})
                os.environ[a.upper()] = str(c[k].attrs[a])

    print('>>> New System Configuration Loaded <<< ')

    return json.dumps(current_settings)


def _create_archive_database():
    """ Create an empty archive.meta.info.db3 database if it doesn't exist

        This file is responsible for tracking the ingest history/progess
        as well as the individual files that have been ingested.
    """
    db_path = os.path.join(TELEMETRY_ARCHIVE, 'archive.meta.info.db3')
    if not os.path.exists(db_path):
        script_path = get_env_variable('JETA_ARCHIVE_DEFINITION_SOURCE')
        with open(script_path, 'r') as s:
            db_definition_script = s.read()
            print(f'Creating archive tracking database (sqlite3) {db_path}')
            db = sqlite3.connect(db_path)
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

    :param msid: the index.h5 file's msid will be created and initialized
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
            idx_file.create_table(idx_file.root, 'epoch', compound_datatype)


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

    path = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/{target}.h5"

    with tables.open_file(path, 'a') as h5:
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
    """Create the msid directory which will store all associated files

    :param msid: the msid for which a directory will be created
    :type msid: str
    """

    msid_directory_path = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/"
    if not os.path.exists(msid_directory_path):
        os.makedirs(msid_directory_path)


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

    :param msid: msid archive to backup. Defaults to 'All'
    :type msid: str, optional
    :param data_only: only backup the *.h5 files, defaults to False
    :type data_only: bool, optional
    """
    pass


def restore(uuid: str):
    """Restore the state of the archive to a particular point

    :param uuid: the uuid of the snapshot to restore
    :type uuid: uuid
    """
    print(uuid)
    pass


def truncate(target_date):
    """Truncate msid and statfiles for every archive file after date (to nearest
    year:doy)

    :param date: threshold data to truncate as a doy object or yday string
    :type date: astropy.time.Time or string date
    """

    td = Time(target_date, format='yday').jd

    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'a') as ref_data:

        all_msid_last_timestamps = []

        for msid in ref_data.keys():
            try:
                # LITA-215: do not truncate if
                # last_ingested_timestamp < target_date
                if ref_data[msid].attrs['last_ingested_timestamp'] < td:
                    continue
            except Exception as err:
                param = 'last_ingested_timestamp'
                print(f"Skipped {msid}, could not read {param}, reason: {err}")
                continue

            # default timestamp last timestamp
            last_ingest_time = get_env_variable('EPOCH')

            idx_path = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/index.h5"

            idx_file = h5py.File(idx_path, mode='a')

            # get a list of all times in the index file
            index_times = idx_file['epoch'][...]['epoch']

            # get the indices of a sub selected list
            # thresholded on the target date
            index_list = np.argwhere(index_times < target_date)

            try:
                # get the last index that meets the critiera
                checkpoint = index_list[-1][0]
                # get the index in both the times and values files
                # that map to a checkpoint
                target_index = idx_file['epoch'][...]['index'][checkpoint]
            except Exception as err:
                idx_file.close()
                print(f'Skipped {msid}, missing checkpoint in {index_list}')
                print(f'reason: \n {err}')
                continue

            idx_file.close()
            value_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/values.h5"
            times_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/times.h5"
            index_filepath = f"{ENG_ARCHIVE}/archive/data/tlm/{msid}/index.h5"

            values_h5 = tables.open_file(value_filepath, mode='a')
            times_h5 = tables.open_file(times_filepath, mode='a')
            idx_file = tables.open_file(index_filepath, mode='a')

            # Do the actual work of truncating
            values_h5.root.values.truncate(target_index)
            times_h5.root.times.truncate(target_index)
            idx_file.root.epoch.truncate(checkpoint)

            try:
                stats_target_time = idx_file.root.epoch[-1][0]
            except Exception as err:
                print('INFO: Defaulting to mission epoch for stats truncate.')
                print(err)
                stats_target_time = get_env_variable('EPOCH')

            time0 = Time(stats_target_time, format='jd').unix

            values_h5.close()
            times_h5.close()
            idx_file.close()

            # Truncate stats
            from jeta.archive.update import del_stats

            try:
                del_stats(msid, time0, '5min')
                del_stats(msid, time0, 'daily')
            except Exception as err:
                print(f"Skipped {msid}, stats truncation, reason: \n{err}")

            try:
                with h5py.File(times_filepath, 'r') as times:
                    last_ingest_time = times['times'][...][-1]
            except Exception as err:
                print(f"Skipping msid {msid}, reason {err}")

            ref_data[msid].attrs['last_ingested_timestamp'] = last_ingest_time
            all_msid_last_timestamps.append(last_ingest_time)

        max_time_stamp = max(all_msid_last_timestamps)
        ref_data.attrs['last_ingested_timestamp'] = max_time_stamp
        print(f"Max Archive Time: {ref_data.attrs['last_ingested_timestamp']}")
        return 0


def destory(data_only=True):
    """Destory the archive by removing all data 

    :param data_only: if True only remove the file data, defaults to True
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
        except FileNotFoundError:
            return "Nothing to do. Archive does not exist."


def add_msid_to_archive(msid, dtype, nrows, nbytes):
    """Add a single msid to the archive by creating the required files and layout

    :param msid: the msid to be added to the archive
    :type msid: str
    :param dtype: the numpy data type of the msid
    :type dtype: np.dtype
    :param nrows: the number of rows expected for the lifetime to the msid
    :type nrows: int
    :param nbytes: the number of bytes used in string representation
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
    system_epoch = get_env_variable('EPOCH')
    with h5py.File(ALL_KNOWN_MSID_METAFILE, 'a') as h5:
        for msid in h5.keys():
            add_msid_to_archive(
                msid,
                dtype=np.float64,
                nrows=calculate_expected_rows(4),
                nbytes=h5[msid].attrs['nbytes']
            )
            # Set the default value to DEC 24 2021, 00:00:00
            h5[msid].attrs['last_ingested_timestamp'] = system_epoch
        h5.attrs['last_ingested_timestamp'] = system_epoch


if __name__ == "__main__":
    initialize()
