import abc

import pandas as pd


class LoadStrategy(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):

        self.filepath = filepath
       
    @abc.abstractmethod
    def execute(self):
        pass


class LoadFlatCSVStrategy(LoadStrategy):

    def __init__(self, filepath):

        super(LoadFlatCSVStrategy, self).__init__(filepath)
        self.filepath = filepath

    def execute(self):
        # load the file from .csv into pandas as DataFrame
        # NOTE: maybe use chunksize to return a TextFileReader instead
        return pd.read_csv(self.filepath, encoding="ISO-8859-1", engine="c")


class LoadHDF5Strategy(LoadStrategy):

    def execute(self):
        print("Loading HDF5")