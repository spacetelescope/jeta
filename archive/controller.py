import os
import sys
import glob
import json
import sqlite3

from jeta.archive.utils import get_env_variable

archive_root=get_env_variable('TELEMETRY_ARCHIVE')
archive_data_area='tlm'
archive_stating_area=get_env_variable('STAGING_DIRECTORY')
archive_recovery_area='recovery'

class Utilities:

    @staticmethod
    def get_list_of_staged_files():
        """Get telemetry files"""

        files = []
        supported_file_types = ['h5']
        staging_directory = get_env_variable('STAGING_DIRECTORY')

        print(f"Starting ingest file discovery in {staging_directory} ... ")

        for file_type in supported_file_types:

            files.extend(sorted(glob.glob(f"{staging_directory}*.{file_type}")))

        print(f"Discovered: {len(files)} in {staging_directory} ...")
        print(f"Files discovered: {files}")

        return files

    @staticmethod
    def prepare_archive_on_disk():
        # create the telemetry data area inside the arhcive root
        telemetry_archive = os.path.join(archive_root, archive_data_area)
        os.makedirs(telemetry_archive, exist_ok=True)

        # Init meta database
        with open(get_env_variable('ARCHIVE_DEFINITION_SOURCE')) as sql_file:
            c = sqlite3.connect(os.path.join(telemetry_archive, 'meta.db'))
            c.executescript(sql_file.read())
            c.commit()
            c.close()

    def __init__(self):
        pass