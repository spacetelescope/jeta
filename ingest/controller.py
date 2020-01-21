import os
import sys
import glob
import json
import sqlite3

from datetime import datetime

import h5py
import tables
import numpy as np


from celery import Task
from celery import Celery
from celery import group
from celery.result import AsyncResult
from celery.schedules import schedule as celery_ingest_schedule
from celery.task.control import inspect

from redbeat.schedulers import RedBeatSchedulerEntry

from jeta.archive.utils import get_env_variable

from jeta.ingest import process
from jeta.config.celery import app
from jeta.ingest.archive import DataProduct

from jeta.archive.controller import Utilities

# FIXME: Path handing management, maybe just use pyyaks of jinja2 directy
archive_root = get_env_variable('TELEMETRY_ARCHIVE')
archive_data_area = 'tlm'
archive_stating_area = get_env_variable('STAGING_DIRECTORY')
archive_recovery_area = 'recovery'

init_index_file = DataProduct.init_mnemonic_index_file
init_values_file = DataProduct.create_values_hdf5
init_times_file = DataProduct.create_times_hdf5

update_super_controller = None


@app.task()
def _execute_ingest_task(ingest_file_list):

    for idx, ingest_filepath in enumerate(ingest_file_list):

        ingest = _load_data_into_memory(ingest_filepath)
        print(f'Loaded data from {ingest_filepath} starting processing')
        _update_mnemonic_index_file(ingest)
        _initialze_mnemonic_filesets(ingest)
        _execute_append_data_subtasks(ingest)


@app.task
def _append_data_to_jeta_archive(mnemonic, data, times):

        try:
            h5 = tables.open_file(f"/Users/dkauffman/Projects/jSka/jeta/data/tlm/{mnemonic}/times.h5", mode='a')
            times_array = np.array(times, dtype='f8')
            h5.root.time.append(times_array)

            h5.close()

            h5 = tables.open_file(f"/Users/dkauffman/Projects/jSka/jeta/data/tlm/{mnemonic}/values.h5", mode='a')
            eu_values_array = np.array(data, dtype='U8')
            h5.root.data.append(eu_values_array)

            h5.close()
        except Exception as err:
            raise ValueError(err.args[0])


def set_ingest_schedule(interval=celery_ingest_schedule(run_every=600)):

    entry = RedBeatSchedulerEntry('execute_telemetry_ingest', 'jeta.ingest.controller._execute_automated_ingest', interval, app=app)
    entry.save()

    return 'The Schedule Update Ran'


def get_mnemonic_init_list(ingest):

    mnemomics_with_filesets = []
    telemetry_archive = os.path.join(archive_root, archive_data_area)

    conn = sqlite3.connect(os.path.join(telemetry_archive, 'meta.db'))
    cursor = conn.cursor()

    mnemomics_with_filesets = cursor.execute('SELECT mnemonic FROM initialized_mnemonics;')
    try:
        mnemomics_with_filesets = list(mnemomics_with_filesets.fetchall())
    except Exception as err:
        print(err.args[0])
        raise
    mnemonics_without_filesets = np.setdiff1d(list(ingest.data.keys()), mnemomics_with_filesets, assume_unique=True)

    conn.close()

    return mnemonics_without_filesets


def init_mnemonic_fileset(mnemonic, ingest):

    # FIXME: Replace hard coded paths with template
    init_values_file(mnemonic, ingest.data, f"/Users/dkauffman/Projects/jSka/jeta/data/tlm/{mnemonic}/values.h5")
    init_times_file(mnemonic, ingest.data, f"/Users/dkauffman/Projects/jSka/jeta/data/tlm/{mnemonic}/times.h5")


def record_mnemonic_as_initialized(mnemonic):

    telemetry_archive = os.path.join(archive_root, archive_data_area)

    conn = sqlite3.connect(os.path.join(telemetry_archive, 'meta.db'))
    cursor = conn.cursor()
    cursor.execute(f"INSERT OR REPLACE INTO initialized_mnemonics VALUES ('{mnemonic}', 1);")

    conn.commit()
    conn.close()


def _load_data_into_memory(ingest_filepath):

    ingest = process.Ingest(ingest_filepath, os.path.join(archive_root, archive_data_area), strategy='h5py').start()

    return ingest


def _update_mnemonic_index_file(ingest):

    for mnemonic in ingest.data.keys():

        init_index_file(archive_root, mnemonic, ingest.data[mnemonic]['index']['index'], ingest.data[mnemonic]['index']['epoch'])


def _initialze_mnemonic_filesets(ingest):

    print(f'INFO {datetime.now()}: Initializing Mnemonic Filesets  ...') # TODO: pass as state message and log

    mnemonics_to_initialize = len(get_mnemonic_init_list(ingest))

    init_list = get_mnemonic_init_list(ingest)

    for idx, mnemonic in enumerate(init_list):
        print(f'INFO: Initalizing {idx + 1}/{mnemonics_to_initialize} mnemonics ...', end="\r")
        init_mnemonic_fileset(mnemonic, ingest)
        record_mnemonic_as_initialized(mnemonic)


def _execute_append_data_subtasks(ingest):

        print(f'INFO: {datetime.now()}: Appending data to archive ...')

        lazy_group = group([_append_data_to_jeta_archive.s(mnemonic, list(ingest.data[mnemonic]['values']), list(ingest.data[mnemonic]['times'])) for mnemonic in ingest.data.keys()])
        result = lazy_group.apply_async()

        return result


def get_ingest_status(task_id):

        inspection = inspect()
        if task_id:
            result = app.AsyncResult(task_id)

            if result.ready():
                end_time = datetime.now()
            else:
                end_time = 'N/A'

            return json.dumps({
                    'complete': result.ready(),
                    'failed': result.failed(),
                    'state': str(result.state),
                    'info': str(result.info),
                    'task_id': str(task_id),
                    'end_time': str(end_time),
                    'task_inspection': inspection.active(),
                })
        return json.dumps({
            'complete': False,
            'state': 'ID NOT IN SYSTEM.',
            'end_time': str(end_time),
            })


@app.task
def execute():

    # Initialize the archive with manifest files.
    Utilities.prepare_archive_on_disk()

    # Get a list of the files to ingest into the archive.
    list_of_ingest_files = Utilities.get_list_of_staged_files()

    """ Build an initial response to the calling function.

        The response includes the start of the ingest process
        The task id to track ingest status
        and the list of files being ingested.
    """
    response = {
        'ingest_start_time': str(datetime.now()),
        'task_id': None,
        'list_of_ingest_files': list_of_ingest_files,
    }

    # Start an async task
    async_result = _execute_ingest_task.delay(list_of_ingest_files)

    response['task_id'] = async_result.task_id

    return response


@app.task
def _execute_automated_ingest():
    from jeta.config.celery import app
    print(app.conf.CELERYBEAT_SCHEDULE)
    execute()
