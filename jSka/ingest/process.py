import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.time import Time

from .strategy import LoadFlatCSVStrategy 
from .strategy import LoadHDF5Strategy

from ..config import properties

class Ingest:

    
    # Pandas Data Frame for storing raw data ingested from an input file.
    df = None

    # The column headers in extracted from the input file
    headers = None

    # The delta times collection
    delta_times =  collections.defaultdict(list)

    load_strategy = {
        'csv': LoadFlatCSVStrategy,
        'hdf5': LoadHDF5Strategy
    }

    def set_delta_times(self, mnemonic):

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

        for idx, row in self.df.iterrows():

            self.values[row[properties.NAME_COLUMN]].append(row[properties.VALUE_COLUMN])
            self.times[row[properties.NAME_COLUMN]].append(str(row[properties.TIME_COLUMN]).replace("/", "-"))
        
        self.headers = list(self.values.keys())
        
        for mnemonic in self.headers:
            self.set_delta_times(mnemonic)

        # NOTE not sure about this, quick and dirty ,
        # maybe move or change altogther
        
        for key, value in self.values.items():
            self.values[key] = np.array(value)

        
        return self.values, self.times
        
    def start(self):

        # load data into Dataframe from some source
        self.df = self._source_import_method.execute()

        self.set_min_entry_date(self.df.iloc[0][properties.TIME_COLUMN])
        self.set_max_entry_date(self.df.iloc[-1][properties.TIME_COLUMN])

        
        # Sort the data into buckets, this will have to be another strategy 
        # since the format of the data will be completely different depending on the 
        # file type ingested. For now flat csv is assumed.
        self.data = self.partition()


        # Create the HDF5 file(s) archive 
        # self.archive()

        return self.data
      
    def __init__(self, input_file, input_path=properties.INGEST_DIR):

        self.input_path=Path(properties.INGEST_DIR)
        self.input_file=input_file
        self.full_input_path=self.input_path.joinpath(self.input_file)
        self._source_import_method = self.load_strategy['csv'](self.full_input_path)
