import os
from pathlib import Path

import numpy as np
import tables
import h5py

from tables import *
import tables3_api

import pyyaks.logger

loglevel = pyyaks.logger.VERBOSE
logger = pyyaks.logger.get_logger(name='jskaarchive', level=loglevel,
                                  format="%(asctime)s %(message)s")


class Epoch(IsDescription):

    index = UInt64Col()
    epoch = Float64Col()


class DataProduct:

    """ This class is responsible for managing the output of the three per mnemonic DataProducts

        see: Telemetry Archive Structure in the documentation for detials about how data is stored.
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
                # logger.verbose('INFO: created new directory {} for column {}'
                #    .format(filedir+"/"+mnemonic, mnemonic))
            except IOError as e:
                raise IOError("Failed to create directory.")

        return str(filedir+"/"+mnemonic)


    @staticmethod
    def get_file_write_path(fullpath, mnemonic, h5type=None):

        """ This static method gets the file write path for one of the three different h5 data products.

        TODO: Full path contains an extra component and will need to be replaced.
        The interafce will remain the same and just accept the path to the parent
        archive directory. Is will be worked using JSKA-35

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
        :param fullpath: the fullpath to the parent archive directory, becomes the full path to the archive file.
        :returns: h5, fullpath: a reference to the h5 file object and the path on disk
        """

        #fullpath = DataProduct.get_file_write_path(fullpath, mnemonic, h5type='values')

        if not os.path.exists(fullpath):

            filters = tables.Filters(complevel=5, complib='zlib')
            h5 = tables.open_file(str(fullpath), driver="H5FD_CORE", mode="a", filters=filters)

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


            h5.create_earray(h5.root, 'data', h5type, h5shape, title=mnemonic,
                        expectedrows=n_rows)


            # logger.verbose('WARNING: made new file {} for column {!r} shape={} with n_rows(1e6)={}'
            #         .format(fullpath, mnemonic, None, None))

            h5.close()

            return h5, fullpath


    @staticmethod
    def create_times_hdf5(mnemonic, data, fullpath):

        #fullpath = DataProduct.get_file_write_path(fullpath, mnemonic, h5type='times')

        if not os.path.exists(fullpath):
            """
                TODO:
                    Ecapsulate This Block, the method is doing to many things and the
                    code will have to be repeated elsewhere anyway.
            """

            #########BLOCK#############
            col = data[mnemonic]
            times = col['times']
            dt = np.median(times[1:] - times[:-1])

            if dt < 1:
                dt = 1.0
            n_rows = int(365 * 20 / dt)

            ##########END BLOCK#########

            filters = tables.Filters(complevel=5, complib='zlib')
            h5 = tables.open_file(str(fullpath), driver="H5FD_CORE", mode="a", filters=filters)

            h5shape = (0,)

            h5timetype = tables.Atom.from_dtype(times.dtype)

            h5.create_earray(h5.root, 'time', h5timetype, h5shape, title='Time',
                        expectedrows=n_rows)

            h5.close()


    @staticmethod
    def get_archive_file_length(parent_directory, mnemonic):

        file_length = 0

        fullpath = f'{os.environ["TELEMETRY_ARCHIVE"]}tlm/{mnemonic}/values.h5'

        if os.path.exists(fullpath):
            h5 = tables.open_file(str(fullpath), driver="H5FD_CORE", mode="r")
            table = h5.root.data
            file_length = len(table)
            h5.close()

        return file_length

    @staticmethod
    def get_last_known_epoch(parent_directory, mnemonic):

        fullpath = DataProduct.get_file_write_path(parent_directory, mnemonic, 'index')

        if os.path.exists(fullpath):
            h5 = tables.open_file(str(fullpath), driver="H5FD_CORE", mode="r")
            table = h5.root.epoch
            last_known_epoch = table[-1][0]
            #print(f'last known epoch {table[-1][1]}')
            h5.close()
        else:
            last_known_epoch = 2454466.5 # 2008-01-01 00:00:00.000 in JD

        #print(f'Last known epoch for this mnemonic is: {last_known_epoch}')
        return last_known_epoch

    @staticmethod
    def init_mnemonic_index_file(archive_root, mnemonic, idx=None, epoch=None):

        fullpath = os.path.join(archive_root, 'tlm', mnemonic, 'index.h5')

        # if os.path.exists(parent_directory):
        #     DataProduct.create_archive_directory(parent_directory, mnemonic)

        filters = tables.Filters(complevel=5, complib='zlib')
        h5 = tables.open_file(str(fullpath), driver="H5FD_CORE", mode="a", filters=filters)

        if idx is not None and epoch is not None:

            try:
                table = h5.create_table(h5.root, 'epoch', Epoch)
            except Exception as err:
                table = h5.root.epoch
            finally:
                table.row['index'] = idx
                table.row['epoch'] = epoch
                table.row.append()
                table.flush()

                # print(table[0][0])

        h5.close()

        # logger.verbose('WARNING: made new file {} for column {!r} shape={} with n_rows(1e6)={}'
        #            .format(fullpath, mnemonic, None, None))

    @staticmethod
    def create_hdf5(mnemonic, data, parent_directory, h5_file_type):

        """ This method may server as a generic replacement to the other two create methods
        """

        fullpath = DataProduct.get_file_write_path(parent_directory, mnemonic, h5_file_type)
        filters = tables.Filters(complevel=5, complib='zlib')
        h5 = tables.open_file(str(fullpath), driver="H5FD_CORE", mode="w", filters=filters)

        n_rows = int(365 * 20)

        #h5.create_earray(h5.root, h5_file_type, title=mnemonic,
        #            expectedrows=n_rows)


        # logger.verbose('WARNING: made new file {} for column {!r} shape={} with n_rows(1e6)={}'
        #            .format(fullpath, mnemonic, None, None))

        h5.close()

    def __init__(self):

        pass
