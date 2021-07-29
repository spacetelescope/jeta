import numba
import numpy as np
import pandas as pd

import h5py
import tables

from jeta.archive.utils import get_env_variable

STAGING_DIRECTORY = get_env_variable('STAGING_DIRECTORY')


@numba.jit(nopython=True)
def _is_sorted_numba(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True

class StagingValidator():

   
    def validate_msid_tracking(self):
        pass


    def validate_ingest_dataset_start_stop(self, h5):
        pass


    def validate_ingest_file_as_consecutive(self, h5file):
        try:
            results = self.validate_ingest_file_datasets_as_consecutive(h5file)
        except Exception as err:
            raise err
        return np.all(results == True)


    def validate_ingest_file_datasets_as_consecutive(self, h5file):
        """ Open an ingest file and verify the time column for
            each dataset is consecutive.
        """
        with h5py.File(f"{STAGING_DIRECTORY}/{h5file}") as h5:
            dataset_labels =  h5['samples'].keys()
            
            results = []
            
            for label in dataset_labels:
                all_dset_times = pd.DataFrame(h5['samples'][label][...])['observatoryTime'][...].to_numpy()
                results.append(np.atleast_1d(_is_sorted_numba(np.atleast_1d(all_dset_times.byteswap().newbyteorder()))))
                
            return np.array(results)


    def validate_consecutive_ingest_file(self, h5):
        pass


    def __init__(self):
        pass
