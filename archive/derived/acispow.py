# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Derived parameter MSIDs related to ACIS power.
"""

import numpy as np
from . import base

class DP_DPA_POWER(base.DerivedParameter):
    """ACIS total DPA-A and DPA-B power"""
    rootparams = ['1dp28avo', '1dpicacu', '1dp28bvo', '1dpicbcu']
    time_step = 32.8
    content_root = 'acispow'

    def calc(self, data):
        power = (data['1dp28avo'].vals * data['1dpicacu'].vals + 
                 data['1dp28bvo'].vals * data['1dpicbcu'].vals)
        return power


class DP_DEA_POWER(base.DerivedParameter):
    """ACIS DEA power"""
    rootparams = ['1de28avo', '1deicacu']
    time_step = 32.8
    content_root = 'acispow'

    def calc(self, data):
        power = data['1de28avo'].vals * data['1deicacu'].vals
        return power


class DP_PSMC_POWER(base.DerivedParameter):
    """ACIS PSMC power"""
    rootparams = ['1dp28avo', '1dpicacu', '1dp28bvo', '1dpicbcu', '1de28avo', '1deicacu']
    time_step = 32.8
    content_root = 'acispow'

    def calc(self, data):
        power = (data['1dp28avo'].vals * data['1dpicacu'].vals + 
                 data['1dp28bvo'].vals * data['1dpicbcu'].vals +
                 data['1de28avo'].vals * data['1deicacu'].vals)
        return power
    
