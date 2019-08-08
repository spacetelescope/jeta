# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Derived parameter MSIDs related to EPS subsystem.

Author: B. Bissell

Revision History::

     Jul 2014 Initial Version
"""

from . import base


class DerivedParameterEps(base.DerivedParameter):
    content_root = 'eps'


#--------------------------------------------
class DP_BATT1_TAVE(DerivedParameterEps):
    """Battery 1 Average Temperature. Derived from average of all three battery temperature sensors.
        Telemetry 16x / MF
    """
    rootparams = ['TB1T1', 'TB1T2', 'TB1T3']
    time_step = 2.05

    def calc(self, data):
        BATT1_TAVE = (data['TB1T1'].vals + data['TB1T2'].vals +
                      data['TB1T3'].vals) / 3
        return BATT1_TAVE


#--------------------------------------------
class DP_BATT2_TAVE(DerivedParameterEps):
    """Battery 2 Average Temperature. Derived from average of all three battery temperature sensors.
        Telemetry 16x / MF
    """
    rootparams = ['TB2T1', 'TB2T2', 'TB2T3']
    time_step = 2.05

    def calc(self, data):
        BATT2_TAVE = (data['TB2T1'].vals + data['TB2T2'].vals +
                      data['TB2T3'].vals) / 3
        return BATT2_TAVE


#--------------------------------------------
class DP_BATT3_TAVE(DerivedParameterEps):
    """Battery 3 Average Temperature. Derived from average of all three battery temperature sensors.
        Telemetry 16x / MF
    """
    rootparams = ['TB3T1', 'TB3T2', 'TB3T3']
    time_step = 2.05

    def calc(self, data):
        BATT3_TAVE = (data['TB3T1'].vals + data['TB3T2'].vals +
                      data['TB3T3'].vals) / 3
        return BATT3_TAVE


#--------------------------------------------
class DP_EPOWER1(DerivedParameterEps):
    """Bus Power = ELBI_LOW * ELBV
        Telemetry 8x / MF
    """
    rootparams = ['ELBI_LOW', 'ELBV']
    time_step = 4.1

    def calc(self, data):
        EPOWER1 = (data['ELBI_LOW'].vals * data['ELBV'].vals)
        return EPOWER1


#--------------------------------------------
class DP_MYSAPOW(DerivedParameterEps):
    """-Y Solar Array Power = ESAMYI * ELBV
        Telemetry 8x / MF
    """
    rootparams = ['ESAMYI', 'ELBV']
    time_step = 4.1

    def calc(self, data):
        MYSAPOW = (data['ESAMYI'].vals * data['ELBV'].vals)
        return MYSAPOW


#--------------------------------------------
class DP_PYSAPOW(DerivedParameterEps):
    """+Y Solar Array Power = ESAPYI * ELBV
        Telemetry 8x / MF
    """
    rootparams = ['ESAPYI', 'ELBV']
    time_step = 4.1

    def calc(self, data):
        PYSAPOW = (data['ESAPYI'].vals * data['ELBV'].vals)
        return PYSAPOW
