import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.time import Time

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
    delta_times =  collections.defaultdict(list)

    # This dictionary contains references to different file loading strategies
    load_strategy = {
        'pandas': LoadPandasCSVStrategy,
        'native': LoadPythonCSVStrategy,
        'pytables': LoadPyTablesHDF5Strategy,
        'h5py': LoadH5PYHDF5Strategy
    }

    data = {}

    def set_delta_times(self, mnemonic, epoch=None):

        times_with_epoch = ['2018-01-01 00:00:00.000']

        times_with_epoch += self.times[mnemonic]

        self.delta_times[mnemonic] = np.diff(Time(times_with_epoch, format='iso', scale='utc').jd)

    def set_ingest_path(self, ingest_path):

        self.input_path = ingest_path

    def set_min_entry_date(self, date_string):

        self.min_entry_date = str(date_string).replace("/", "-")
       
    def set_max_entry_date(self, date_string):

        self.max_entry_date = str(date_string).replace("/", "-")


    def get_delta_times(self, mnemonic):

        return self.delta_times[mnemonic]

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

    def partition(self):
        
        self.values = collections.defaultdict(list)
        self.times =  collections.defaultdict(list)
        self.tstart = None

        for idx, row in self.df.iterrows():

            if self.tstart is None:
                self.tstart = Time(row[properties.TIME_COLUMN].replace("/", "-")).jd

            date = str(row[properties.TIME_COLUMN]).replace("/", "-")
            value = row[properties.VALUE_COLUMN]

            self.values[row[properties.NAME_COLUMN]].append(value)
            self.times[row[properties.NAME_COLUMN]].append(str(date))

        self.tstop = Time(date, format='iso').jd
        
        self.headers = list(self.values.keys())
        
        for mnemonic in self.headers:
            self.set_delta_times(mnemonic)
        
        for mnemonic, value in self.values.items():

            self.data[mnemonic] = {
                'times': Time(self.times[mnemonic], format='iso', in_subfmt='date_hms').jd,
                'values': np.array(self.values[mnemonic])
            }

     
        return self
        
    def start(self):

        # load data into Dataframe from some source
        self.df = self._source_import_method.execute()

        self.set_min_entry_date(self.df.iloc[0][properties.TIME_COLUMN])
        self.set_max_entry_date(self.df.iloc[-1][properties.TIME_COLUMN])

        
        # Sort the data into buckets, this will have to be another strategy 
        # since the format of the data will be completely different depending on the 
        # file type ingested. For now flat csv is assumed.
        self.partition()


        # Create the HDF5 file(s) archive 
        # self.archive()

        return self
      
    def __init__(self, input_file, input_path=properties.INGEST_DIR):

        self.input_path=Path(properties.INGEST_DIR)
        self.input_file=input_file
        self.full_input_path=self.input_path.joinpath(self.input_file)
        self._source_import_method = self.load_strategy['pandas'](self.full_input_path)
