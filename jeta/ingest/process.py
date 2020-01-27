import os
import sys
import collections
from datetime import datetime

from pathlib import Path

import numpy as np
import pandas as pd

from astropy.time import Time

from .archive import DataProduct
from .archive import ROOT_DIR

from .strategy import LoadPandasCSVStrategy
from .strategy import LoadPythonCSVStrategy
from .strategy import LoadPyTablesHDF5Strategy
from .strategy import LoadH5PYHDF5Strategy

from ..config import properties



class Ingest:

    """ This class is responsible for reading and processing input files.

    """

    # Pandas Data Frame for storing raw data ingested from an input file.
    df = None

    # The column headers in extracted from the input file
    headers = None

    # The delta times collection
    delta_times = collections.defaultdict(list)

    # This dictionary contains references to different file loading strategies
    load_strategy = {
        'pandas': LoadPandasCSVStrategy,
        'native': LoadPythonCSVStrategy,
        'pytables': LoadPyTablesHDF5Strategy,
        'h5py': LoadH5PYHDF5Strategy
    }

    data = {}

    values = collections.defaultdict(list)
    times =  collections.defaultdict(list)
    indices = collections.defaultdict(dict)

    tstart = None
    epoch_date = None

    ####### Might move this
    def create_archive_directories(self):

        ingest_mnemonics = np.array(list(self.df.keys()))
        existing_archive_directories = np.array([x[1] for x in os.walk(ROOT_DIR)][0])

        directories_to_create = np.setdiff1d(ingest_mnemonics, existing_archive_directories)

        print("INFO: creating archive directories ... ")

        for archive_subdirectory in directories_to_create:

            try:
                os.makedirs(ROOT_DIR+"/"+archive_subdirectory)
            except IOError as e:
                raise IOError("Failed to create directory.")

    def get_delta_times(self, mnemonic, epoch=None):

            if epoch is None:
                epoch = self.epoch_date

            jd_times = Time(self.times[mnemonic], format='iso', in_subfmt='date_hms').jd

            return np.diff(np.insert(jd_times, 0, self.time_to_quadtime(epoch)))

    def init_times(self, row):

        if self.tstart is None:

            start_time = row[properties.TIME_COLUMN].replace("/", "-")
            self.epoch_date = start_time[:11]+"00:00:00.000"
            self.tstart = Time(start_time).jd

    def set_ingest_path(self, ingest_path):

        self.input_path = ingest_path

    def set_min_entry_date(self, date_string):

        self.min_entry_date = str(date_string).replace("/", "-")

    def set_max_entry_date(self, date_string):

        self.max_entry_date = str(date_string).replace("/", "-")

    def get_times(self, mnemonic):

        return self.times[mnemonic]

    def get_data(self):

        return self.values

    def get_mnemonic_data(self, mnemonic):

        return self.data[mnemonic]

    def get_min_max_year_for_mnemonic(self, mnemonic):

        return [self.times[mnemonic][0][0:4], self.times[mnemonic][-1][0:4]]

    def get_max_entry_date(self):
        pass

    def get_min_entry_date(self):

        pass

    def time_to_quadtime(self, time):

        time = Time(time[:11]+"00:00:00.000").jd
        return time

    def is_new_epoch_required(self, proposed_epoch):

        proposed_epoch = datetime.strptime(proposed_epoch, "%Y-%m-%d %H:%M:%S.%f")
        tmp_epoch = datetime.strptime(self.epoch_date , "%Y-%m-%d %H:%M:%S.%f")

        return proposed_epoch > tmp_epoch

    def update_epoch_time(self, proposed_epoch):

        self.epoch_date = max([proposed_epoch, self.epoch_date])

        return self.epoch_date

    # TODO: Add benchmark decorator
    def process_hdf5_ingest_file(self):

        print("INFO: ingesting mnemonics into memory ...")

        for mnemonic in self.df:

            # NOTE: If data is properly typed this decoding step may not be required,
            # its possible that a different storage decision could be though

            # out_times = self.df[mnemonic]['data']['date']
            # self.values[mnemonic] = self.df[mnemonic]['data']['value'][()]
            # self.times[mnemonic] = [x.decode("utf-8").replace("/", "-") for x in self.df[mnemonic]['data']['date']

            self.values[mnemonic] = [x.decode("utf-8") for x in self.df[mnemonic]['data']['value']]
            self.times[mnemonic] = [x.decode("utf-8").replace("/", "-") for x in self.df[mnemonic]['data']['date']]

        import operator
        self.tstart = Time(self.times[min(self.times.items(), key=operator.itemgetter(1))[0]][0], format='iso').jd
        self.tstop = Time(self.times[max(self.times.items(), key=operator.itemgetter(1))[0]][-1], format='iso').jd

        for mnemonic, value in self.values.items():

            self.times[mnemonic] = sorted(self.times[mnemonic])

            if self.time_to_quadtime(self.times[mnemonic][-1]) == self.time_to_quadtime(self.times[mnemonic][0]):
                index = DataProduct.get_archive_file_length(self.output_path, mnemonic)
                epoch = self.times[mnemonic][0]
                self.indices[mnemonic] = {'index': index, 'epoch': self.time_to_quadtime(epoch)}
            else:
                # This case is reserved for data that is ingested out of sequence.
                pass

            self.data[mnemonic] = {
                'times': self.get_delta_times(mnemonic, epoch),
                'values': np.array(self.values[mnemonic]),
                'index': self.indices[mnemonic],
                'parent_directory': f"{ROOT_DIR}/{mnemonic}"
            }

        return self

    def partition(self):

        self.values = collections.defaultdict(list)
        self.times =  collections.defaultdict(list)
        self.indices = collections.defaultdict(dict)

        for idx, row in self.df.iterrows():

            mnemonic = row[properties.NAME_COLUMN]

            self.init_times(row)

            date = str(row[properties.TIME_COLUMN]).replace("/", "-")

            value = row[properties.VALUE_COLUMN]

            self.values[mnemonic].append(value)
            self.times[mnemonic].append(str(date))

        self.tstop = Time(date, format='iso').jd


        for mnemonic, value in self.values.items():

            self.times[mnemonic] = sorted(self.times[mnemonic])

            parent_directory = DataProduct.create_archive_directory(self.output_path, mnemonic)

            if self.time_to_quadtime(self.times[mnemonic][-1]) == self.time_to_quadtime(self.times[mnemonic][0]):
                index = DataProduct.get_archive_file_length(self.output_path, mnemonic)
                epoch = self.times[mnemonic][0]
                self.indices[mnemonic] = {'index': index, 'epoch': self.time_to_quadtime(epoch)}
            else:
                pass

            self.data[mnemonic] = {
                'times': self.get_delta_times(mnemonic, epoch),
                'values': np.array(self.values[mnemonic]),
                'index': self.indices[mnemonic],
                'parent_directory': parent_directory
            }

        return self

    def start(self):

        # load data into Dataframe from some source
        self.df = self._source_import_method.execute()

        # This wont work because date/times are not in order
        # self.set_min_entry_date(self.df.iloc[0][properties.TIME_COLUMN])
        # self.set_max_entry_date(self.df.iloc[-1][properties.TIME_COLUMN])

        self.create_archive_directories()

        # Sort the data into buckets, this will have to be another strategy
        # since the format of the data will be completely different depending on the
        # file type ingested. For now flat csv is assumed.
        if self.strategy == 'pandas':
            self.partition()
        else:
            self.process_hdf5_ingest_file()


        # Create the HDF5 file(s) archive
        # self.archive()
        # raise ValueError('Done in Error.')
        return self

    def __init__(self, input_file, output_path, strategy='pandas', input_path=properties.STAGING_DIRECTORY):

        self.strategy = strategy
        self.input_path=Path(properties.STAGING_DIRECTORY)
        self.input_file=input_file
        self.output_path = output_path
        self.full_input_path=self.input_path.joinpath(self.input_file)
        self._source_import_method = self.load_strategy[strategy](self.full_input_path)
        print(f"Initialized Ingest Strategy: {self._source_import_method}")
