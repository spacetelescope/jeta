import os
import h5py

from bisect import bisect_left, bisect_right

import numpy as np

class IndexFile:

    def _get_start_index(self, tstart):
        """ Returns the index of the epoch (timestamp) <= the given start time.
        """
        i = bisect_right(self.data['epoch'], tstart)
        if i and not (i-1 < 0):
            return i - 1
        else:
            return 0
        raise ValueError(f'{tstart} did not match any value.')
    
    def _get_stop_index(self, tstop):
        """ Returns the index of the epoch (timestamp) >= the given stop time.
        """
        i = bisect_right(self.data['epoch'], tstop)
        if not (i >= self.data.shape[0]):
            return i
        else:
            return -1
        raise ValueError(f'{tstop} did not match any value.')
    
    def get_index_range(self, tstart, tstop):
        """ Returns a tuple of indices given an start and stop time range.
        """
        return (self._get_start_index(tstart), self._get_stop_index(tstop))

    def get_archive_index_range(self, tstart, tstop):
        idx0 = self.data['index'][self._get_start_index(tstart)]
        idxN = self.data['index'][self._get_stop_index(tstop)]
        return (idx0, idxN)

    
    def get_data(self, tstart=None, tstop=None):
        """ Returns a either the full data for the file or a block given a range.
        """
        if tstart is None or tstop is None:
            return self.data
        else:
            idx0, idxN = self.get_index_range(tstart, tstop)
            return self.data[idx0:idxN]
        
    def __init__(self, msid):
        self.msid = msid
        self.file_path = '{}/data/tlm/{}/index.h5'.format(os.environ['TELEMETRY_ARCHIVE'], msid)
        assert os.path.exists(self.file_path), f'Could not create index file reference. File path does not exist: {self.file_path}'
        with h5py.File(self.file_path, 'r') as h5:
            self.data = h5['epoch'][...]
            self.data = np.array([row for row in self.data], dtype=([('epoch', np.float64), ('index', np.int32)]))
        self.idx0 = self.data['index'][0]
        self.idxN = self.data['index'][-1]
        self.epoch0 = self.data['epoch'][0]
        self.epochN = self.data['epoch'][-1]


class ValueFile:

    def get_file_data_range(self, tstart=None, tstop=None):

        if tstart is None or tstop is None:
            idx0 = 0
            idxN = -1
        else:
            idx0, idxN = self.index_file.get_archive_index_range(tstart, tstop)
        with h5py.File(self.file_path, 'r') as h5:
            self.selection = h5['values'][idx0:idxN]
            self.selection_length = len(self.selection)
            self.length = len(h5['values'])
    
    def __init__(self, msid):
        self.msid = msid
        self.file_path = self.file_path = '{}/data/tlm/{}/values.h5'.format(os.environ['TELEMETRY_ARCHIVE'], msid)
        assert os.path.exists(self.file_path), f'Could not create values file reference. File path does not exist: {self.file_path}'
        self.index_file = IndexFile(self.msid)
        self.selection = None


class TimeFile:

    def get_file_data_range(self, tstart=None, tstop=None):
        if tstart is None and tstop is None:
            idx0 = 0
            idxN = -1
        else:
            idx0, idxN = self.index_file.get_archive_index_range(tstart, tstop)
        with h5py.File(self.file_path, 'r') as h5:
            self.selection = h5['times'][idx0:idxN]
            self.selection_length = len(self.selection)
            self.length = len(h5['times'])
    
    def __init__(self, msid):
        self.msid = msid
        self.file_path = self.file_path = '{}/data/tlm/{}/times.h5'.format(os.environ['TELEMETRY_ARCHIVE'], msid)
        assert os.path.exists(self.file_path), f'Could not create values file reference. File path does not exist: {self.file_path}'
        self.index_file = IndexFile(self.msid)
        self.selection = None
