from pathlib import Path

#import h5py
import tables
from tables import *

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

    def append(self, entry):

        pass

    def create(self):

        pass

    @staticmethod
    def create_values_hdf5(mnemonic, data, filepath):

        n_rows = int(86400 * 365 * 20 / 18)

        filters = tables.Filters(complevel=5, complib='zlib')

        filedir = os.path.dirname(filepath)

        filepath = Path(filedir).joinpath(str(mnemonic+'_values_.h5'))
    
        h5file = tables.open_file(filepath, driver="H5FD_CORE", mode="w", filters=filters)

        col = data[-1][mnemonic]

        h5shape = (0,) + col.shape[1:]

        h5type = tables.Atom.from_dtype(col.dtype)

        h5file.create_earray(h5file.root, 'data', h5type, h5shape, title=mnemonic,
                        expectedrows=n_rows)

    
        h5file.close()

        return filepath

    def create_t0_hdf(self):
        pass
        #print("CREATING: ", self.mu+"_t0.h5" )

    def create_dt_XXXX(self):

        # TODO: variable path, year discovery
        filters = tables.Filters(complevel=5, complib='zlib')
        h5file = tables.open_file("../output/"+self.mu+"_dt_2019.h5", driver="H5FD_CORE", mode="w", filters=filters)
        group = h5file.create_group("/", self.mu, self.mu+' Data')
        table = h5file.create_table(group, 'deltatime', DeltaTime, "DeltaTime")

        x = table.row

        for i in self.deltatime:
            x['delta_time'] = i
            x.append()
        
        table.flush()
        h5file.close()

        #print("CREATING: ", self.mu+"_dt_XXXX.h5" )
    
    def create(self):
        
        #print("OUTPUT DIR: ", self.output_path)

        self.create_values_hdf()
        self.create_t0_hdf()
        self.create_dt_XXXX()
    
    def __init__(self, mu, eu, deltatime, output_path=Path.cwd()):

        self.mu = mu
        self.eu = eu
        self.deltatime = deltatime
        self.output_path=Path(output_path)
        #self.output_file=output_file
#self.full_output_path=self.output_path.joinpath(self.output_file)