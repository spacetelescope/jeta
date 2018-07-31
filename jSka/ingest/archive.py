import os
import numpy as np

from pathlib import Path

import tables
from tables import *
import tables3_api

import pyyaks.logger


loglevel = pyyaks.logger.VERBOSE
logger = pyyaks.logger.get_logger(name='jskaarcive', level=loglevel,
                                  format="%(asctime)s %(message)s")

class DataProduct:

    """ This class is responsible for managing the output of the various DataProduct
    
    """


    @staticmethod 
    def create_archive_directory(fullpath, mnemonic):

        """ This static method creates the archive directory for a specified
        mnemonic. The method body will likely change, but interface will
        remain.

        :param fullpath: this is the path to the archive root directory.
        :param mnemonic: mnemonic
        :raise IOError: raises an IO Error if the directory cannot be created. 
        """
    
        filedir = os.path.dirname(fullpath)

        if not os.path.exists(filedir+"/"+mnemonic):
            try:
                os.makedirs(filedir+"/"+mnemonic)
                logger.verbose('INFO: created new directory {} for column {}'
                   .format(filedir+"/"+mnemonic, mnemonic))
            except IOError as e:
                raise IOError("Failed to create directory.")
           

    @staticmethod
    def get_file_write_path(fullpath, mnemonic, h5type=None):

        """ This static method gets the file write path for one of the three different h5 data products. 
        
        NOTE: As the code is written today full path contains an extra component, 
        a file named  <mnemonic>.h5 this will need to be replaced. The interafce will 
        remain the same and just accept the path to the parent archive directory.

        :param fullpath: this is the fullpath to the parent archive directory.
        :param mnenmonic: mnemonic
        :param h5type: the type (name) of the .h5 file path to return
        :returns: fullpath including file name of the archive file. 
        """

        filetype = {
            'values': 'values.h5',
            'times': 'times.h5',
            'index': 'index.h5',
        }

        if h5type is not None:

            filename = fullpath
            filedir = os.path.dirname(filename)
        
            filepath = Path(filedir+"/"+mnemonic).joinpath(filetype[h5type])

        else: 
            raise ValueError('Error: Invalid filetype. Must be values, times or index.')

        return filepath

    @staticmethod
    def create_values_hdf5(mnemonic, data, fullpath):

        """ This method does the work of creating the hdf5 file that store values.

        :param mnemonic: the mnemonic for which to create a file.
        :param data: the data associated with that mnemonic
        :param fullpath: the fullpath to the parent archive directory, becomes the
        full path to the archive file.
        :returns h5, fullpath: a reference to the h5 file object and the path on disk

        """

        fullpath = DataProduct.get_file_write_path(fullpath, mnemonic, h5type='values')

        filters = tables.Filters(complevel=5, complib='zlib')
        h5 = tables.open_file(fullpath, driver="H5FD_CORE", mode="w", filters=filters)
        
        """ 
            TODO: 
                Ecapsulate This Block, the method is doing to many things and the
                code will have to be repated elsewhere anyway. 
        """
        #########BLOCK#############
        col = data[mnemonic]
        times = col['times']
        values = col['values']
        dt = np.median(times[1:] - times[:-1])

        if dt < 1:
            dt = 1.0
        n_rows = int(365 * 20 / dt)

        ##########END BLOCK#########
    
        h5shape = (0,)
        h5type = tables.Atom.from_dtype(values.dtype)
        #h5timetype = tables.Atom.from_dtype(times.dtype)

        h5.create_earray(h5.root, 'data', h5type, h5shape, title=mnemonic,
                     expectedrows=n_rows)

        # h5.create_earray(h5.root, 'time', h5timetype, h5shape, title='Time',
        #              expectedrows=n_rows)

        logger.verbose('WARNING: made new file {} for column {!r} shape={} with n_rows(1e6)={}'
                   .format(fullpath, mnemonic, None, None))
    
        h5.close()

        return h5, fullpath
    

    @staticmethod
    def create_times_hdf5(mnemonic, data,filepath):

        pass
        
    @staticmethod
    def create_index_hdf5(mnemonic, data,filepath):

        pass

    def __init__(self):

        pass