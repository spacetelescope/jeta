from pathlib import Path

#import h5py
import tables
from tables import *
import tables3_api

import numpy as np
import pyyaks.logger

import os

class DeltaTime(IsDescription):

    delta_time = Float64Col()

class Value(IsDescription):

    eu_values = StringCol(16)

class DataProduct:

    output_directory = None
    

    def calc_n_rows(self):

        n_rows = int(86400 * 365 * 20 / 18)
        
        return n_rows

    @staticmethod
    def get_archive_path(mnemonic, filedir):

        filedir = os.path.dirname(filedir)
        filepath = Path(filedir).joinpath(str(mnemonic+'_values.h5'))

        return filepath

    @staticmethod
    def get_file_write_path(fullpath, mnemonic, type):

        filename = fullpath
        filedir = os.path.dirname(filename)
       
        filepath = Path(filedir+"/"+mnemonic).joinpath(str('values.h5'))

        return filepath

    @staticmethod
    def create_values_hdf5(mnemonic, data, filepath):

        filepath = Path(filepath).joinpath(str('values.h5'))

        filters = tables.Filters(complevel=5, complib='zlib')
        h5 = tables.open_file(filepath, driver="H5FD_CORE", mode="w", filters=filters)

        col = data[mnemonic]
        times = col['times']
        values = col['values']
        dt = np.median(times[1:] - times[:-1])

        if dt < 1:
            dt = 1.0
            print(dt)
        n_rows = int(365 * 20 / dt)
    
        h5shape = (0,)
        h5type = tables.Atom.from_dtype(values.dtype)
        h5timetype = tables.Atom.from_dtype(times.dtype)

        h5.create_earray(h5.root, 'data', h5type, h5shape, title=mnemonic,
                     expectedrows=n_rows)

        h5.create_earray(h5.root, 'time', h5timetype, h5shape, title='Time',
                     expectedrows=n_rows)

    
        h5.close()

        return h5, filepath
    
    @staticmethod
    def append_delta_times(mnemonic, delta_times, filepath):
        
        filters = tables.Filters(complevel=5, complib='zlib')

        h5file = tables.open_file(filepath, driver="H5FD_CORE", mode="w", filters=filters)
        group = h5file.create_group("/", mnemonic, "Data")
        table = h5file.create_table(group, 'deltatime', DeltaTime, "DeltaTime")

        row = table.row

        for delta_time in delta_times:
            row['delta_time'] = delta_time
            row.append()
        
        table.flush()
        h5file.close()

    @staticmethod
    def create_delta_time_hdf5(mnemonic, filepath):

        filedir = os.path.dirname(filepath)
        delta_time_filepath = Path(filedir).joinpath(str(mnemonic+'_delta_times.h5'))

        filters = tables.Filters(complevel=5, complib='zlib')

        h5file = tables.open_file(filepath, driver="H5FD_CORE", mode="w", filters=filters)
      
        h5file.close()

        return delta_time_filepath

    def __init__(self, mu, eu, deltatime, output_path=Path.cwd()):

        self.mu = mu
        self.eu = eu
        self.deltatime = deltatime
        self.output_path=Path(output_path)
