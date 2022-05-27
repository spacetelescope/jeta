# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:08:42 2022

@author: ahunter
"""

import pandas as pd
import h5py
import sys
import numpy as np
from jeta.archive import operations
from jeta.archive.utils import get_env_variable

epoch = 2459572.5
np_dtype = 'np.float64'

ALL_KNOWN_MSID_METAFILE = get_env_variable('ALL_KNOWN_MSID_METAFILE')

if not sys.argv[1].endswith('.xlsx'):
    print ('bad argument')
    sys.exit()

prd = pd.read_excel(sys.argv[1], sheet_name=0, skiprows=3)

with h5py.File(ALL_KNOWN_MSID_METAFILE, 'a') as h5:
    print (f'starting length: {len(h5.keys())}')
    msid_list = list(h5.keys())

    for index, row in prd.iterrows():
        mnemonic = row['TlmMnemonic']
        tlmid = int(row['TlmIdentifier'])
        prd_dtype = row['TlmParameterDataType']
        nbytes = 8
        
        if mnemonic not in msid_list:
            print (f'adding {mnemonic}')
            
            h5.create_group(mnemonic)
            h5[mnemonic].attrs['id'] = tlmid
            h5[mnemonic].attrs['last_ingested_timestamp'] = epoch
            h5[mnemonic].attrs['nbytes'] = nbytes
            h5[mnemonic].attrs['numpy_datatype'] = np_dtype
            h5[mnemonic].attrs['parameter_datatype'] = prd_dtype
            
            
            operations.add_msid_to_archive(
                mnemonic, 
                dtype=np.float64,  
                nrows=operations.calculate_expected_rows(4),
                nbytes=nbytes
            )
            

    print (f'ending length: {len(h5.keys())}')
