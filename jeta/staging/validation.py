import numba
import numpy as np
import pandas as pd

import h5py
import tables

from jeta.archive.utils import get_env_variable

STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')


class StagingValidator():

    @numba.jit(nopython=True)
    def _is_sorted_numba(a):
        for i in range(a.size-1):
            if a[i+1] < a[i]:
                return False
        return True


    def validate_msid_tracking(self):
        pass


    def validate_ingest_dataset_start_stop(self, h5):
        pass


    def validate_ingest_file_as_consecutive(self, h5file):
        results = self.validate_ingest_file_datasets_as_consecutive(self, h5file)
        return np.all(results == True)


    def validate_ingest_file_datasets_as_consecutive(self, h5file):
        """ Open an ingest file and verify the time column for
            each dataset is consecutive.
        """
        h5 = h5py.File(h5file)
        
        dataset_labels =  h5['samples'].keys()
        
        results = []
        
        for label in dataset_labels:
            all_dset_times = pd.DataFrame(h5['samples'][label][...])['observatoryTime'][...].to_numpy()
            results.append(self._is_sorted_numba(all_dset_times.byteswap().newbyteorder()))
            
        h5.close()
        return np.array(results)


    def validate_consecutive_ingest_file(self, h5):
        pass


    def __init__(self):
        pass
