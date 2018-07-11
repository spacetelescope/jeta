import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import LoadFlatCSVStrategy 
from .strategy import LoadHDF5Strategy

from ..config import properties

class Ingest:

    
    # Pandas Data Frame for storing raw data ingested from an input file.
    df = None

    # The column headers in extracted from the input file
    headers = None

    load_strategy = {
        'csv': LoadFlatCSVStrategy,
        'hdf5': LoadHDF5Strategy
    }

    def get_data(self):

        return self.values

    def get_mnemonic_data(self, mnemonic):

        return self.data[mnemonic]

    def set_ingest_path(self, ingest_path):

        self.input_path = ingest_path

    def create_values_hdf(self, data, filename):

            pass

            # filters = tables.Filters(complevel=5, complib='zlib')
            # h5file = tables.open_file(filename, driver="H5FD_CORE", mode="w", filters=filters)
            # group = h5file.create_group("/", self.mu, self.mu+' Data')
            # table = h5file.create_table(group, 'values', Value, "EU Values")

            # h5file.close()

    def partition(self):
        
        self.values = collections.defaultdict(list)
        self.times =  collections.defaultdict(list)

        for idx, row in self.df.iterrows():

            self.values[row[properties.NAME_COLUMN]].append(row[properties.VALUE_COLUMN])
            self.times[row[properties.NAME_COLUMN]].append(row[properties.TIME_COLUMN])

   
        self.headers = list(self.values.keys())
        

        # NOTE not sure about this, quick and dirty ,
        # maybe move or change altogther
        
        for key, value in self.values.items():
            self.values[key] = np.array(value)

        
        return self.values, self.times
        
    def start(self):

        # load data into Dataframe from some source
        self.df = self._source_import_method.execute()

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
