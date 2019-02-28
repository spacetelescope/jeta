import abc

import h5py
import pandas as pd


from ..core.exceptions import StrategyNotImplemented


class LoadStrategy(object):

    """ This is an abstract base class to encapsulate a file loading strategy
        for loading an ingest from from disk.

        Possible strategies include:
        - Loading a CVS using Pandas
        - Loading a CSV using native python file open
        - Loading a HDF5 file using pytables
        - Loading a HDF5 file using h5py

        NOTE: Currently only one strategy, loading a CSV file using pandas
        is implemented.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):

        self.filepath = filepath

    @abc.abstractmethod
    def execute(self):
        pass


class LoadPandasCSVStrategy(LoadStrategy):

    def __str__(self):

        return "LoadPandasCSVStrategy: Ingest FOF (.csv) files using pandas read_csv."

    def __init__(self, filepath):

        super(LoadPandasCSVStrategy, self).__init__(filepath)
        self.filepath = filepath

    def execute(self):
        # load the file from .csv into pandas as DataFrame
        # NOTE: maybe use chunksize to return a TextFileReader instead
        return pd.read_csv(self.filepath, encoding="ISO-8859-1", engine="c")

class LoadPythonCSVStrategy(LoadStrategy):

    def execute(self):
        # TODO: Add LOGGING
        print("Loading CVS using native Python.")
        raise StrategyNotImplemented('No implementation for loading an HDF5 file using pytables')

class LoadPyTablesHDF5Strategy(LoadStrategy):

    def execute(self):
        # TODO: Add LOGGING
        print("Loading HDF5 using PyTables.")
        raise StrategyNotImplemented('No implementation for loading an HDF5 file using pytables')


class LoadH5PYHDF5Strategy(LoadStrategy):

    def __str__(self):

        return "LoadH5PYHDF5Strategy: Ingest FOF (.h5) files using pandas h5py.File."

    def __init__(self, filepath):

        super(LoadH5PYHDF5Strategy, self).__init__(filepath)
        self.filepath = filepath

    def execute(self):

        h5 = h5py.File(self.filepath)

        print("Loading HDF5 using h5py.")

        return h5