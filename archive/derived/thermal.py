# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Derived parameter MSIDs related to thermal subsystems.
"""

import numpy as np

from . import base


class DerivedParameterThermal(base.DerivedParameter):
    content_root = 'thermal'

    def fix_4OHTRZ50(self, data):
        # This is the first period when zone 50 became stuck on, up until it
        # became temporarily unstuck in 2006:363.
        stuck1 = (data.times > 188439810.759) & (data.times < 283792685.184)
        # Zone 50 became stuck again on day 2006:364.
        stuck2 = data.times > 283844996.184
        stuck = stuck1 | stuck2
        data['4OHTRZ50'].vals[stuck] = 1

#--------------------------------------------


class DP_EE_AXIAL(DerivedParameterThermal):
    rootparams = ['OHRTHR58', 'OHRTHR12', 'OHRTHR36', 'OHRTHR56', 'OHRTHR57',
                  'OHRTHR55', 'OHRTHR35', 'OHRTHR37', 'OHRTHR34', 'OHRTHR13',
                  'OHRTHR10', 'OHRTHR11']
    time_step = 32.8

    def calc(self, data):
        HYPAVE = (data['OHRTHR12'].vals + data['OHRTHR13'].vals +
                  data['OHRTHR36'].vals + data['OHRTHR37'].vals +
                  data['OHRTHR57'].vals + data['OHRTHR58'].vals) / 6
        PARAVE = (data['OHRTHR10'].vals + data['OHRTHR11'].vals +
                  data['OHRTHR34'].vals + data['OHRTHR35'].vals +
                  data['OHRTHR55'].vals + data['OHRTHR56'].vals) / 6
        HAAG = PARAVE - HYPAVE
        DTAXIAL = np.abs(1.0 * HAAG)
        EE_AXIAL = DTAXIAL * 0.0034
        return EE_AXIAL


#--------------------------------------------
class DP_EE_BULK(DerivedParameterThermal):
    rootparams = ['OHRTHR10', 'OHRTHR58', 'OHRTHR52', 'OHRTHR53', 'OHRTHR56',
                  'OHRTHR57', 'OHRTHR54', 'OHRTHR55', 'OHRTHR12', 'OHRTHR35',
                  'OHRTHR11', 'OHRTHR08', 'OHRTHR09', 'OHRTHR31', 'OHRTHR33',
                  'OHRTHR34', 'OHRTHR13', 'OHRTHR36', 'OHRTHR37']
    time_step = 32.8

    def calc(self, data):
        P_SUM = (data['OHRTHR10'].vals + data['OHRTHR11'].vals +
                 data['OHRTHR34'].vals + data['OHRTHR35'].vals +
                 data['OHRTHR55'].vals + data['OHRTHR56'].vals)
        H_SUM = (data['OHRTHR12'].vals + data['OHRTHR13'].vals +
                 data['OHRTHR36'].vals + data['OHRTHR37'].vals +
                 data['OHRTHR57'].vals + data['OHRTHR58'].vals)
        CAP_SUM = (data['OHRTHR08'].vals + data['OHRTHR09'].vals +
                   data['OHRTHR31'].vals + data['OHRTHR33'].vals +
                   data['OHRTHR52'].vals + data['OHRTHR53'].vals +
                   data['OHRTHR54'].vals)
        HMCSAVE = (CAP_SUM + P_SUM + H_SUM) / 19.0
        DTBULK = np.abs(1.0 * HMCSAVE - 69.8)
        EE_BULK = DTBULK * 0.0267

        return EE_BULK


#--------------------------------------------
class DP_EE_DIAM(DerivedParameterThermal):
    """Kodak diametrical encircled energy"""
    rootparams = ['OHRMGRD6', 'OHRMGRD3']
    time_step = 32.8

    def calc(self, data):
        VAL2 = np.abs(1.0 * data['OHRMGRD6'].vals)
        VAL1 = np.abs(1.0 * data['OHRMGRD3'].vals)
        DTDIAM = np.max([VAL1, VAL2], axis=0)
        EE_DIAM = DTDIAM * 0.401
        return EE_DIAM


#--------------------------------------------
class DP_EE_RADIAL(DerivedParameterThermal):
    rootparams = ['OHRTHR52', 'OHRTHR53', 'OHRTHR54', 'OHRTHR31', 'OHRTHR09',
                  'OHRTHR08', 'OHRTHR33']
    time_step = 32.8

    def calc(self, data):
        CAPIAVE = (data['OHRTHR09'].vals + data['OHRTHR53'].vals +
                   data['OHRTHR54'].vals) / 3
        CAPOAVE = (data['OHRTHR08'].vals + data['OHRTHR31'].vals +
                   data['OHRTHR33'].vals + data['OHRTHR52'].vals) / 4
        HARG = CAPOAVE - CAPIAVE
        DTRADIAL = np.abs(1.0 * HARG)
        EE_RADIAL = DTRADIAL * 0.0127
        return EE_RADIAL


#--------------------------------------------
class DP_EE_THERM(DerivedParameterThermal):
    rootparams = ['OHRTHR37', 'OHRTHR58', 'OHRMGRD6', 'OHRMGRD3', 'OHRTHR35',
                  'OHRTHR52', 'OHRTHR53', 'OHRTHR56', 'OHRTHR57', 'OHRTHR54',
                  'OHRTHR55', 'OHRTHR12', 'OHRTHR36', 'OHRTHR08', 'OHRTHR09',
                  'OHRTHR31', 'OHRTHR33', 'OHRTHR34', 'OHRTHR13', 'OHRTHR10',
                  'OHRTHR11']
    time_step = 32.8

    def calc(self, data):
        CAP_SUM = (data['OHRTHR08'].vals + data['OHRTHR09'].vals +
                   data['OHRTHR31'].vals + data['OHRTHR33'].vals +
                   data['OHRTHR52'].vals + data['OHRTHR53'].vals +
                   data['OHRTHR54'].vals)
        CAPIAVE = (data['OHRTHR09'].vals + data['OHRTHR53'].vals +
                   data['OHRTHR54'].vals) / 3
        P_SUM = (data['OHRTHR10'].vals + data['OHRTHR11'].vals +
                 data['OHRTHR34'].vals + data['OHRTHR35'].vals +
                 data['OHRTHR55'].vals + data['OHRTHR56'].vals)
        CAPOAVE = (data['OHRTHR08'].vals + data['OHRTHR31'].vals +
                   data['OHRTHR33'].vals + data['OHRTHR52'].vals) / 4
        H_SUM = (data['OHRTHR12'].vals + data['OHRTHR13'].vals +
                 data['OHRTHR36'].vals + data['OHRTHR37'].vals +
                 data['OHRTHR57'].vals + data['OHRTHR58'].vals)
        PARAVE = (data['OHRTHR10'].vals + data['OHRTHR11'].vals +
                  data['OHRTHR34'].vals + data['OHRTHR35'].vals +
                  data['OHRTHR55'].vals + data['OHRTHR56'].vals) / 6
        HYPAVE = (data['OHRTHR12'].vals + data['OHRTHR13'].vals +
                  data['OHRTHR36'].vals + data['OHRTHR37'].vals +
                  data['OHRTHR57'].vals + data['OHRTHR58'].vals) / 6
        VAL1 = np.abs(1.0 * data['OHRMGRD3'].vals)
        VAL2 = np.abs(1.0 * data['OHRMGRD6'].vals)

        HAAG = PARAVE - HYPAVE
        HARG = CAPOAVE - CAPIAVE
        HMCSAVE = (CAP_SUM + P_SUM + H_SUM) / 19.0

        DTAXIAL = np.abs(1.0 * HAAG)
        DTRADIAL = np.abs(1.0 * HARG)
        DTDIAM = np.max([VAL1, VAL2], axis=0)
        DTBULK = np.abs(1.0 * HMCSAVE - 69.8)

        EE_RADIAL = DTRADIAL * 0.0127
        EE_AXIAL = DTAXIAL * 0.0034
        EE_BULK = DTBULK * 0.0267
        EE_DIAM = DTDIAM * 0.401
        EE_THERM = (EE_BULK + EE_AXIAL + EE_RADIAL + EE_DIAM)

        return EE_THERM


#--------------------------------------------
class DP_HAAG(DerivedParameterThermal):
    rootparams = ['OHRTHR58', 'OHRTHR12', 'OHRTHR56', 'OHRTHR57', 'OHRTHR55',
                  'OHRTHR13', 'OHRTHR36', 'OHRTHR37', 'OHRTHR34', 'OHRTHR35',
                  'OHRTHR10', 'OHRTHR11']
    time_step = 32.8

    def calc(self, data):
        HYPAVE = (data['OHRTHR12'].vals + data['OHRTHR13'].vals +
                  data['OHRTHR36'].vals + data['OHRTHR37'].vals +
                  data['OHRTHR57'].vals + data['OHRTHR58'].vals) / 6
        PARAVE = (data['OHRTHR10'].vals + data['OHRTHR11'].vals +
                  data['OHRTHR34'].vals + data['OHRTHR35'].vals +
                  data['OHRTHR55'].vals + data['OHRTHR56'].vals) / 6
        HAAG = PARAVE - HYPAVE
        return HAAG


#--------------------------------------------
class DP_HARG(DerivedParameterThermal):
    rootparams = ['OHRTHR52', 'OHRTHR53', 'OHRTHR54', 'OHRTHR31', 'OHRTHR09',
                  'OHRTHR08', 'OHRTHR33']
    time_step = 32.8

    def calc(self, data):
        CAPIAVE = (data['OHRTHR09'].vals + data['OHRTHR53'].vals +
                   data['OHRTHR54'].vals) / 3
        CAPOAVE = (data['OHRTHR08'].vals + data['OHRTHR31'].vals +
                   data['OHRTHR33'].vals + data['OHRTHR52'].vals) / 4
        HARG = CAPOAVE - CAPIAVE
        return HARG


#--------------------------------------------
class DP_HMAX35(DerivedParameterThermal):
    rootparams = ['OHRTHR52', 'OHRTHR53', 'OHRTHR50', 'OHRTHR51', 'OHRTHR56',
                  'OHRTHR55', 'OHRTHR23', 'OHRTHR22', 'OHRTHR30', 'OHRTHR33',
                  'OHRTHR12', 'OHRTHR13', 'OHRTHR10', 'OHRTHR11', 'OHRTHR36',
                  'OHRTHR37', 'OHRTHR49', 'OHRTHR45', 'OHRTHR44', 'OHRTHR47',
                  'OHRTHR46', 'OHRTHR42', 'OHRTHR29', 'OHRTHR02', 'OHRTHR05',
                  'OHRTHR04', 'OHRTHR07', 'OHRTHR06', 'OHRTHR09', 'OHRTHR08',
                  'OHRTHR21', 'OHRTHR27', 'OHRTHR26', 'OHRTHR25', 'OHRTHR24',
                  'OHRTHR03']
    time_step = 32.8

    def calc(self, data):
        HMAX35 = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            HMAX35 = np.max([HMAX35, data[names].vals], axis=0)
        return HMAX35


#--------------------------------------------
class DP_HMCSAVE(DerivedParameterThermal):
    rootparams = ['OHRTHR10', 'OHRTHR58', 'OHRTHR52', 'OHRTHR53', 'OHRTHR56',
                  'OHRTHR57', 'OHRTHR54', 'OHRTHR55', 'OHRTHR12', 'OHRTHR35',
                  'OHRTHR11', 'OHRTHR08', 'OHRTHR09', 'OHRTHR31', 'OHRTHR33',
                  'OHRTHR34', 'OHRTHR13', 'OHRTHR36', 'OHRTHR37']
    time_step = 32.8

    def calc(self, data):
        P_SUM = (data['OHRTHR10'].vals + data['OHRTHR11'].vals +
                 data['OHRTHR34'].vals + data['OHRTHR35'].vals +
                 data['OHRTHR55'].vals + data['OHRTHR56'].vals)
        H_SUM = (data['OHRTHR12'].vals + data['OHRTHR13'].vals +
                 data['OHRTHR36'].vals + data['OHRTHR37'].vals +
                 data['OHRTHR57'].vals + data['OHRTHR58'].vals)
        CAP_SUM = (data['OHRTHR08'].vals + data['OHRTHR09'].vals +
                   data['OHRTHR31'].vals + data['OHRTHR33'].vals +
                   data['OHRTHR52'].vals + data['OHRTHR53'].vals +
                   data['OHRTHR54'].vals)
        HMCSAVE = (CAP_SUM + P_SUM + H_SUM) / 19.0
        return HMCSAVE


#--------------------------------------------
class DP_HMIN35(DerivedParameterThermal):
    rootparams = ['OHRTHR52', 'OHRTHR53', 'OHRTHR50', 'OHRTHR51', 'OHRTHR56',
                  'OHRTHR55', 'OHRTHR23', 'OHRTHR08', 'OHRTHR30', 'OHRTHR33',
                  'OHRTHR12', 'OHRTHR13', 'OHRTHR36', 'OHRTHR11', 'OHRTHR10',
                  'OHRTHR37', 'OHRTHR49', 'OHRTHR45', 'OHRTHR44', 'OHRTHR47',
                  'OHRTHR46', 'OHRTHR42', 'OHRTHR29', 'OHRTHR02', 'OHRTHR05',
                  'OHRTHR04', 'OHRTHR07', 'OHRTHR06', 'OHRTHR09', 'OHRTHR22',
                  'OHRTHR21', 'OHRTHR27', 'OHRTHR26', 'OHRTHR25', 'OHRTHR24',
                  'OHRTHR03']
    time_step = 32.8

    def calc(self, data):
        HMIN35 = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            HMIN35 = np.min([HMIN35, data[names].vals], axis=0)
        return HMIN35


#--------------------------------------------
class DP_HRMA_AVE(DerivedParameterThermal):
    rootparams = ['OHRTHR52', 'OHRTHR53', 'OHRTHR50', 'OHRTHR51', 'OHRTHR56',
                  'OHRTHR55', 'OHRTHR09', 'OHRTHR08', 'OHRTHR30', 'OHRTHR33',
                  'OHRTHR12', 'OHRTHR13', 'OHRTHR10', 'OHRTHR11', 'OHRTHR36',
                  'OHRTHR37', 'OHRTHR49', 'OHRTHR45', 'OHRTHR44', 'OHRTHR47',
                  'OHRTHR46', 'OHRTHR42', 'OHRTHR29', 'OHRTHR02', 'OHRTHR05',
                  'OHRTHR04', 'OHRTHR07', 'OHRTHR06', 'OHRTHR23', 'OHRTHR22',
                  'OHRTHR21', 'OHRTHR27', 'OHRTHR26', 'OHRTHR25', 'OHRTHR24',
                  'OHRTHR03']
    time_step = 32.8

    def calc(self, data):
        HSUM = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            HSUM = HSUM + data[names].vals
        HRMA_AVE = HSUM / 36
        return HRMA_AVE


#--------------------------------------------
class DP_HRMHCHK(DerivedParameterThermal):
    rootparams = ['OHRTHR52', 'OHRTHR53', 'OHRTHR50', 'OHRTHR51', 'OHRTHR56',
                  'OHRTHR55', 'OHRTHR09', 'OHRTHR08', 'OHRTHR30', 'OHRTHR33',
                  'OHRTHR12', 'OHRTHR13', 'OHRTHR10', 'OHRTHR11', 'OHRTHR36',
                  'OHRTHR37', 'OHRTHR49', 'OHRTHR45', 'OHRTHR44', 'OHRTHR47',
                  'OHRTHR46', 'OHRTHR42', 'OHRTHR03', 'OHRTHR02', 'OHRTHR05',
                  'OHRTHR04', 'OHRTHR07', 'OHRTHR06', 'OHRTHR23', 'OHRTHR22',
                  'OHRTHR21', 'OHRTHR27', 'OHRTHR26', 'OHRTHR25', 'OHRTHR24',
                  'OHRTHR29']
    time_step = 32.8

    def calc(self, data):
        HMIN35 = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            HMIN35 = np.min([HMIN35, data[names].vals], axis=0)
        HMAX35 = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            HMAX35 = np.max([HMAX35, data[names].vals], axis=0)
        HRMHCHK = HMAX35 - HMIN35
        return HRMHCHK


#--------------------------------------------
class DP_OBAAG(DerivedParameterThermal):
    rootparams = ['4RT704T', '4RT705T', '4RT708T', '4RT707T', '4RT709T', '4RT711T',
                  '4RT700T', '4RT702T', '4RT701T', '4RT703T', 'OOBTHR34',
                  'OOBTHR33', 'OOBTHR31', 'OOBTHR62', 'OOBTHR63', '4RT706T',
                  '4RT710T']
    time_step = 32.8

    def calc(self, data):
        AVE2 = (data['4RT705T'].vals + data['4RT706T'].vals +
                data['4RT707T'].vals + data['4RT708T'].vals +
                data['4RT709T'].vals + data['4RT710T'].vals +
                data['4RT711T'].vals)
        AVE1 = (data['OOBTHR62'].vals + data['OOBTHR63'].vals +
                data['4RT700T'].vals + data['4RT701T'].vals +
                data['4RT702T'].vals + data['4RT703T'].vals +
                data['4RT704T'].vals)
        DIAVE = (data['OOBTHR31'].vals + data['OOBTHR33'].vals +
                 data['OOBTHR34'].vals) / 3
        AXAVE = (AVE1 + AVE2) / 14
        OBAAG = AXAVE - DIAVE
        return OBAAG


#--------------------------------------------
class DP_OBAAGW(DerivedParameterThermal):
    rootparams = ['4RT705T', '4RT707T', '4RT709T', '4RT711T', '4RT701T', '4RT703T',
                  'OOBTHR34', 'OOBTHR33', 'OOBTHR31']
    time_step = 32.8

    def calc(self, data):
        AFT_FIT = (data['OOBTHR31'].vals + data['OOBTHR33'].vals +
                   data['OOBTHR34'].vals) / 3
        FWD_FIT = (data['4RT701T'].vals + data['4RT703T'].vals +
                   data['4RT705T'].vals + data['4RT707T'].vals +
                   data['4RT709T'].vals + data['4RT711T'].vals) / 6
        OBAAGW = FWD_FIT - AFT_FIT
        return OBAAGW


#--------------------------------------------
class DP_OBACAVE(DerivedParameterThermal):
    rootparams = ['OOBTHR19', 'OOBTHR18', 'OOBTHR15', 'OOBTHR14', 'OOBTHR17',
                  'OOBTHR11', 'OOBTHR10', 'OOBTHR13', 'OOBTHR12', 'OOBTHR30',
                  'OOBTHR08', 'OOBTHR09', 'OOBTHR24', 'OOBTHR25', 'OOBTHR26',
                  'OOBTHR27', 'OOBTHR20', 'OOBTHR21', 'OOBTHR22', 'OOBTHR23',
                  'OOBTHR28', 'OOBTHR29']
    time_step = 32.8

    def calc(self, data):
        MIDCONE = (data['OOBTHR19'].vals + data['OOBTHR20'].vals +
                   data['OOBTHR21'].vals + data['OOBTHR22'].vals +
                   data['OOBTHR23'].vals + data['OOBTHR24'].vals +
                   data['OOBTHR25'].vals)
        AFTCONE = (data['OOBTHR26'].vals + data['OOBTHR27'].vals +
                   data['OOBTHR28'].vals + data['OOBTHR29'].vals +
                   data['OOBTHR30'].vals)
        FWDCONE = (data['OOBTHR08'].vals + data['OOBTHR09'].vals +
                   data['OOBTHR10'].vals + data['OOBTHR11'].vals +
                   data['OOBTHR12'].vals + data['OOBTHR13'].vals +
                   data['OOBTHR14'].vals + data['OOBTHR15'].vals +
                   data['OOBTHR17'].vals + data['OOBTHR18'].vals)
        OBACAVE = (FWDCONE + MIDCONE + AFTCONE) / 22
        return OBACAVE


#--------------------------------------------
class DP_OBACAVEW(DerivedParameterThermal):
    rootparams = ['4RT705T', 'OOBTHR19', '4RT707T', 'OOBTHR15', 'OOBTHR14',
                  '4RT711T', 'OOBTHR11', 'OOBTHR10', 'OOBTHR13', '4RT701T',
                  'OOBTHR34', 'OOBTHR33', 'OOBTHR31', 'OOBTHR30', 'OOBTHR18',
                  '4RT709T', '4RT703T', 'OOBTHR17', 'OOBTHR08', 'OOBTHR09',
                  'OOBTHR24', 'OOBTHR25', 'OOBTHR26', 'OOBTHR27', 'OOBTHR20',
                  'OOBTHR21', 'OOBTHR22', 'OOBTHR23', 'OOBTHR12', 'OOBTHR28',
                  'OOBTHR29']
    time_step = 32.8

    def calc(self, data):
        FWD_FIT = (data['4RT701T'].vals + data['4RT703T'].vals +
                   data['4RT705T'].vals + data['4RT707T'].vals +
                   data['4RT709T'].vals + data['4RT711T'].vals) / 6
        AFTCONE = (data['OOBTHR26'].vals + data['OOBTHR27'].vals +
                   data['OOBTHR28'].vals + data['OOBTHR29'].vals +
                   data['OOBTHR30'].vals)
        AFT_FIT = (data['OOBTHR31'].vals + data['OOBTHR33'].vals +
                   data['OOBTHR34'].vals) / 3
        MIDCONE = (data['OOBTHR19'].vals + data['OOBTHR20'].vals +
                   data['OOBTHR21'].vals + data['OOBTHR22'].vals +
                   data['OOBTHR23'].vals + data['OOBTHR24'].vals +
                   data['OOBTHR25'].vals)
        FWDCONE = (data['OOBTHR08'].vals + data['OOBTHR09'].vals +
                   data['OOBTHR10'].vals + data['OOBTHR11'].vals +
                   data['OOBTHR12'].vals + data['OOBTHR13'].vals +
                   data['OOBTHR14'].vals + data['OOBTHR15'].vals +
                   data['OOBTHR17'].vals + data['OOBTHR18'].vals)
        OBACAVE = (FWDCONE + MIDCONE + AFTCONE) / 22
        OBACAVEW = (OBACAVE * 148. - FWD_FIT * 70. - AFT_FIT * 29.) / 49.
        return OBACAVEW


#--------------------------------------------
class DP_OBADIG(DerivedParameterThermal):
    rootparams = ['OOBTHR08', 'OOBTHR19', 'OOBTHR31', 'OOBTHR13', 'OOBTHR26',
                  'OOBTHR34', 'OOBTHR33', 'OOBTHR22', 'OOBTHR23', 'OOBTHR60',
                  'OOBTHR61', 'OOBTHR28', 'OOBTHR29']
    time_step = 32.8

    def calc(self, data):
        MZSAVE = (data['OOBTHR08'].vals + data['OOBTHR19'].vals +
                  data['OOBTHR26'].vals + data['OOBTHR31'].vals +
                  data['OOBTHR60'].vals) / 5
        PZSAVE = (data['OOBTHR13'].vals + data['OOBTHR22'].vals +
                  data['OOBTHR23'].vals + data['OOBTHR28'].vals +
                  data['OOBTHR29'].vals + data['OOBTHR61'].vals +
                  data['OOBTHR33'].vals + data['OOBTHR34'].vals) / 8
        OBADIG = MZSAVE - PZSAVE
        return OBADIG


#--------------------------------------------
class DP_OBADIGW(DerivedParameterThermal):
    rootparams = ['OOBTHR08', '4RT705T', 'OOBTHR19', '4RT707T', 'OOBTHR22',
                  '4RT711T', 'OOBTHR13', '4RT701T', 'OOBTHR26', 'OOBTHR34',
                  'OOBTHR33', 'OOBTHR31', 'OOBTHR23', 'OOBTHR60', 'OOBTHR61',
                  'OOBTHR28', 'OOBTHR29']
    time_step = 32.8

    def calc(self, data):
        FWD_FIT_PZ = (data['4RT705T'].vals + data['4RT707T'].vals) / 2. * 70.0
        AFT_FIT_MZ = data['OOBTHR31'].vals * 29.
        PZSAVE = (data['OOBTHR13'].vals + data['OOBTHR22'].vals +
                  data['OOBTHR23'].vals + data['OOBTHR28'].vals +
                  data['OOBTHR29'].vals + data['OOBTHR61'].vals +
                  data['OOBTHR33'].vals + data['OOBTHR34'].vals) / 8
        MZSAVE = (data['OOBTHR08'].vals + data['OOBTHR19'].vals +
                  data['OOBTHR26'].vals + data['OOBTHR31'].vals +
                  data['OOBTHR60'].vals) / 5
        AFT_FIT_PZ = (data['OOBTHR33'].vals + data['OOBTHR34'].vals) / 2 * 29.
        FWD_FIT_MZ = (data['4RT701T'].vals + data['4RT711T'].vals) / 2. * 70.0
        OBADIG = MZSAVE - PZSAVE
        OBADIGW = (OBADIG * 148. - (FWD_FIT_MZ - FWD_FIT_PZ) -
                   (AFT_FIT_MZ - AFT_FIT_PZ)) / 49.
        return OBADIGW


#--------------------------------------------
class DP_OBA_AVE(DerivedParameterThermal):
    rootparams = ['OOBTHR19', 'OOBTHR18', 'OOBTHR15', 'OOBTHR14', 'OOBTHR17',
                  'OOBTHR11', 'OOBTHR10', 'OOBTHR13', 'OOBTHR12', 'OOBTHR37',
                  'OOBTHR36', 'OOBTHR35', 'OOBTHR34', 'OOBTHR33', 'OOBTHR31',
                  'OOBTHR30', 'OOBTHR39', 'OOBTHR38', 'OOBTHR08', 'OOBTHR09',
                  'OOBTHR24', 'OOBTHR25', 'OOBTHR26', 'OOBTHR27', 'OOBTHR20',
                  'OOBTHR21', 'OOBTHR22', 'OOBTHR23', 'OOBTHR46', 'OOBTHR44',
                  'OOBTHR45', 'OOBTHR28', 'OOBTHR29', 'OOBTHR40', 'OOBTHR41']
    time_step = 32.8

    def calc(self, data):
        OSUM = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            OSUM = OSUM + data[names].vals
        OBA_AVE = OSUM / 35
        return OBA_AVE


#--------------------------------------------
class DP_OMAX34(DerivedParameterThermal):
    rootparams = ['OOBTHR19', 'OOBTHR18', 'OOBTHR15', 'OOBTHR14', 'OOBTHR17',
                  'OOBTHR11', 'OOBTHR10', 'OOBTHR13', 'OOBTHR12', 'OOBTHR37',
                  'OOBTHR36', 'OOBTHR35', 'OOBTHR34', 'OOBTHR33', 'OOBTHR31',
                  'OOBTHR30', 'OOBTHR39', 'OOBTHR38', 'OOBTHR28', 'OOBTHR08',
                  'OOBTHR09', 'OOBTHR24', 'OOBTHR25', 'OOBTHR26', 'OOBTHR27',
                  'OOBTHR20', 'OOBTHR21', 'OOBTHR22', 'OOBTHR23', 'OOBTHR46',
                  'OOBTHR45', 'OOBTHR42', 'OOBTHR29', 'OOBTHR40', 'OOBTHR41']
    time_step = 32.8

    def calc(self, data):
        OMAX34 = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            OMAX34 = np.max([OMAX34, data[names].vals], axis=0)
        return OMAX34


#--------------------------------------------
class DP_OMIN34(DerivedParameterThermal):
    rootparams = ['OOBTHR19', 'OOBTHR18', 'OOBTHR15', 'OOBTHR14', 'OOBTHR17',
                  'OOBTHR11', 'OOBTHR10', 'OOBTHR13', 'OOBTHR12', 'OOBTHR37',
                  'OOBTHR36', 'OOBTHR35', 'OOBTHR34', 'OOBTHR33', 'OOBTHR31',
                  'OOBTHR30', 'OOBTHR39', 'OOBTHR38', 'OOBTHR28', 'OOBTHR08',
                  'OOBTHR09', 'OOBTHR24', 'OOBTHR25', 'OOBTHR26', 'OOBTHR27',
                  'OOBTHR20', 'OOBTHR21', 'OOBTHR22', 'OOBTHR23', 'OOBTHR46',
                  'OOBTHR45', 'OOBTHR42', 'OOBTHR29', 'OOBTHR40', 'OOBTHR41']
    time_step = 32.8

    def calc(self, data):
        OMIN34 = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            OMIN34 = np.min([OMIN34, data[names].vals], axis=0)
        return OMIN34


#--------------------------------------------
class DP_P01(DerivedParameterThermal):
    """Zone 1 heater power"""
    rootparams = ['ELBV', '4OHTRZ01']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P01 = data['4OHTRZ01'].vals * VSQUARED / 110.2
        return P01


#--------------------------------------------
class DP_P02(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ02']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P02 = data['4OHTRZ02'].vals * VSQUARED / 109.7
        return P02


#--------------------------------------------
class DP_P03(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ03']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P03 = data['4OHTRZ03'].vals * VSQUARED / 109.4
        return P03


#--------------------------------------------
class DP_P04(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ04']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P04 = data['4OHTRZ04'].vals * VSQUARED / 175.9
        return P04


#--------------------------------------------
class DP_P05(DerivedParameterThermal):
    rootparams = ['4OHTRZ05', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P05 = data['4OHTRZ05'].vals * VSQUARED / 175.7
        return P05


#--------------------------------------------
class DP_P06(DerivedParameterThermal):
    rootparams = ['4OHTRZ06', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P06 = data['4OHTRZ06'].vals * VSQUARED / 175.6
        return P06


#--------------------------------------------
class DP_P07(DerivedParameterThermal):
    rootparams = ['4OHTRZ07', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P07 = data['4OHTRZ07'].vals * VSQUARED / 135.8
        return P07


#--------------------------------------------
class DP_P08(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ08']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P08 = data['4OHTRZ08'].vals * VSQUARED / 36.1
        return P08


#--------------------------------------------
class DP_P09(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ09']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P09 = data['4OHTRZ09'].vals * VSQUARED / 32.6
        return P09


#--------------------------------------------
class DP_P10(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ10']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P10 = data['4OHTRZ10'].vals * VSQUARED / 34.9
        return P10


#--------------------------------------------
class DP_P11(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ11']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P11 = data['4OHTRZ11'].vals * VSQUARED / 39.4
        return P11


#--------------------------------------------
class DP_P12(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ12']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P12 = data['4OHTRZ12'].vals * VSQUARED / 40.3
        return P12


#--------------------------------------------
class DP_P13(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ13']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P13 = data['4OHTRZ13'].vals * VSQUARED / 39.7
        return P13


#--------------------------------------------
class DP_P14(DerivedParameterThermal):
    rootparams = ['4OHTRZ14', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P14 = data['4OHTRZ14'].vals * VSQUARED / 41.2
        return P14


#--------------------------------------------
class DP_P15(DerivedParameterThermal):
    rootparams = ['4OHTRZ15', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P15 = data['4OHTRZ15'].vals * VSQUARED / 40.5
        return P15


#--------------------------------------------
class DP_P16(DerivedParameterThermal):
    rootparams = ['4OHTRZ16', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P16 = data['4OHTRZ16'].vals * VSQUARED / 41.3
        return P16


#--------------------------------------------
class DP_P17(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ17']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P17 = data['4OHTRZ17'].vals * VSQUARED / 116.0
        return P17


#--------------------------------------------
class DP_P18(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ18']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P18 = data['4OHTRZ18'].vals * VSQUARED / 115.7
        return P18


#--------------------------------------------
class DP_P19(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ19']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P19 = data['4OHTRZ19'].vals * VSQUARED / 95.3
        return P19


#--------------------------------------------
class DP_P20(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ20']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P20 = data['4OHTRZ20'].vals * VSQUARED / 379.0
        return P20


#--------------------------------------------
class DP_P23(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ23']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P23 = data['4OHTRZ23'].vals * VSQUARED / 386.0
        return P23


#--------------------------------------------
class DP_P24(DerivedParameterThermal):
    rootparams = ['4OHTRZ24', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P24 = data['4OHTRZ24'].vals * VSQUARED / 385.8
        return P24


#--------------------------------------------
class DP_P25(DerivedParameterThermal):
    rootparams = ['4OHTRZ25', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P25 = data['4OHTRZ25'].vals * VSQUARED / 383.0
        return P25


#--------------------------------------------
class DP_P26(DerivedParameterThermal):
    rootparams = ['4OHTRZ26', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P26 = data['4OHTRZ26'].vals * VSQUARED / 383.5
        return P26


#--------------------------------------------
class DP_P27(DerivedParameterThermal):
    rootparams = ['4OHTRZ27', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P27 = data['4OHTRZ27'].vals * VSQUARED / 383.0
        return P27


#--------------------------------------------
class DP_P28(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ28']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P28 = data['4OHTRZ28'].vals * VSQUARED / 382.3
        return P28


#--------------------------------------------
class DP_P29(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ29']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P29 = data['4OHTRZ29'].vals * VSQUARED / 384.0
        return P29


#--------------------------------------------
class DP_P30(DerivedParameterThermal):
    rootparams = ['4OHTRZ30', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P30 = data['4OHTRZ30'].vals * VSQUARED / 383.0
        return P30


#--------------------------------------------
class DP_P31(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ31']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P31 = data['4OHTRZ31'].vals * VSQUARED / 32.2
        return P31


#--------------------------------------------
class DP_P32(DerivedParameterThermal):
    rootparams = ['4OHTRZ32', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P32 = data['4OHTRZ32'].vals * VSQUARED / 28.6
        return P32


#--------------------------------------------
class DP_P33(DerivedParameterThermal):
    rootparams = ['4OHTRZ33', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P33 = data['4OHTRZ33'].vals * VSQUARED / 36.9
        return P33


#--------------------------------------------
class DP_P34(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ34']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P34 = data['4OHTRZ34'].vals * VSQUARED / 28.0
        return P34


#--------------------------------------------
class DP_P35(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ35']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P35 = data['4OHTRZ35'].vals * VSQUARED / 32.2
        return P35


#--------------------------------------------
class DP_P36(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ36']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P36 = data['4OHTRZ36'].vals * VSQUARED / 44.3
        return P36


#--------------------------------------------
class DP_P37(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ37']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P37 = data['4OHTRZ37'].vals * VSQUARED / 32.1
        return P37


#--------------------------------------------
class DP_P38(DerivedParameterThermal):
    rootparams = ['4OHTRZ38', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P38 = data['4OHTRZ38'].vals * VSQUARED / 27.8
        return P38


#--------------------------------------------
class DP_P39(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ39']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P39 = data['4OHTRZ39'].vals * VSQUARED / 36.8
        return P39


#--------------------------------------------
class DP_P40(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ40']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P40 = data['4OHTRZ40'].vals * VSQUARED / 28.3
        return P40


#--------------------------------------------
class DP_P41(DerivedParameterThermal):
    rootparams = ['4OHTRZ41', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P41 = data['4OHTRZ41'].vals * VSQUARED / 61.7
        return P41


#--------------------------------------------
class DP_P42(DerivedParameterThermal):
    rootparams = ['4OHTRZ42', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P42 = data['4OHTRZ42'].vals * VSQUARED / 51.7
        return P42


#--------------------------------------------
class DP_P43(DerivedParameterThermal):
    rootparams = ['4OHTRZ43', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P43 = data['4OHTRZ43'].vals * VSQUARED / 36.8
        return P43


#--------------------------------------------
class DP_P44(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ44']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P44 = data['4OHTRZ44'].vals * VSQUARED / 36.9
        return P44


#--------------------------------------------
class DP_P45(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ45']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P45 = data['4OHTRZ45'].vals * VSQUARED / 36.8
        return P45


#--------------------------------------------
class DP_P46(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ46']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P46 = data['4OHTRZ46'].vals * VSQUARED / 36.5
        return P46


#--------------------------------------------
class DP_P47(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ47']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P47 = data['4OHTRZ47'].vals * VSQUARED / 52.3
        return P47


#--------------------------------------------
class DP_P48(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ48']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P48 = data['4OHTRZ48'].vals * VSQUARED / 79.5
        return P48


#--------------------------------------------
class DP_P49(DerivedParameterThermal):
    rootparams = ['4OHTRZ49', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P49 = data['4OHTRZ49'].vals * VSQUARED / 34.8
        return P49


#--------------------------------------------
class DP_P50(DerivedParameterThermal):
    rootparams = ['4OHTRZ50', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        self.fix_4OHTRZ50(data)
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P50 = data['4OHTRZ50'].vals * VSQUARED / 35.2
        return P50


#--------------------------------------------
class DP_P51(DerivedParameterThermal):
    rootparams = ['4OHTRZ51', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P51 = data['4OHTRZ51'].vals * VSQUARED / 35.4
        return P51


#--------------------------------------------
class DP_P52(DerivedParameterThermal):
    rootparams = ['4OHTRZ52', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P52 = data['4OHTRZ52'].vals * VSQUARED / 34.4
        return P52


#--------------------------------------------
class DP_P53(DerivedParameterThermal):
    rootparams = ['4OHTRZ53', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P53 = data['4OHTRZ53'].vals * VSQUARED / 94.1
        return P53


#--------------------------------------------
class DP_P54(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ54']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P54 = data['4OHTRZ54'].vals * VSQUARED / 124.4
        return P54


#--------------------------------------------
class DP_P55(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ55']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P55 = data['4OHTRZ55'].vals * VSQUARED / 126.8
        return P55


#--------------------------------------------
class DP_P57(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ57']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P57 = data['4OHTRZ57'].vals * VSQUARED / 142.3
        return P57


#--------------------------------------------
class DP_P58(DerivedParameterThermal):
    rootparams = ['4OHTRZ58', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P58 = data['4OHTRZ58'].vals * VSQUARED / 83.7
        return P58


#--------------------------------------------
class DP_P59(DerivedParameterThermal):
    rootparams = ['4OHTRZ59', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P59 = data['4OHTRZ59'].vals * VSQUARED / 29.7
        return P59


#--------------------------------------------
class DP_P60(DerivedParameterThermal):
    rootparams = ['4OHTRZ60', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P60 = data['4OHTRZ60'].vals * VSQUARED / 30.7
        return P60


#--------------------------------------------
class DP_P61(DerivedParameterThermal):
    rootparams = ['4OHTRZ61', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P61 = data['4OHTRZ61'].vals * VSQUARED / 33.7
        return P61


#--------------------------------------------
class DP_P62(DerivedParameterThermal):
    rootparams = ['4OHTRZ62', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P62 = data['4OHTRZ62'].vals * VSQUARED / 36.1
        return P62


#--------------------------------------------
class DP_P63(DerivedParameterThermal):
    rootparams = ['4OHTRZ63', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P63 = data['4OHTRZ63'].vals * VSQUARED / 36.1
        return P63


#--------------------------------------------
class DP_P64(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ64']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P64 = data['4OHTRZ64'].vals * VSQUARED / 44.1
        return P64


#--------------------------------------------
class DP_P65(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ65']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P65 = data['4OHTRZ65'].vals * VSQUARED / 37.5
        return P65


#--------------------------------------------
class DP_P66(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ66']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P66 = data['4OHTRZ66'].vals * VSQUARED / 29.8
        return P66


#--------------------------------------------
class DP_P67(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ67']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P67 = data['4OHTRZ67'].vals * VSQUARED / 52.0
        return P67


#--------------------------------------------
class DP_P68(DerivedParameterThermal):
    rootparams = ['4OHTRZ68', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P68 = data['4OHTRZ68'].vals * VSQUARED / 29.0
        return P68


#--------------------------------------------
class DP_P69(DerivedParameterThermal):
    rootparams = ['4OHTRZ69', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P69 = data['4OHTRZ69'].vals * VSQUARED / 37.5
        return P69


#--------------------------------------------
class DP_P75(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ75']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P75 = data['4OHTRZ75'].vals * VSQUARED / 130.2
        return P75


#--------------------------------------------
class DP_P76(DerivedParameterThermal):
    rootparams = ['4OHTRZ76', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P76 = data['4OHTRZ76'].vals * VSQUARED / 133.4
        return P76


#--------------------------------------------
class DP_P77(DerivedParameterThermal):
    rootparams = ['4OHTRZ77', 'ELBV']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P77 = data['4OHTRZ77'].vals * VSQUARED / 131.5
        return P77


#--------------------------------------------
class DP_P78(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ78']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P78 = data['4OHTRZ78'].vals * VSQUARED / 133.2
        return P78


#--------------------------------------------
class DP_P79(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ79']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P79 = data['4OHTRZ79'].vals * VSQUARED / 133.1
        return P79


#--------------------------------------------
class DP_P80(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ80']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P80 = data['4OHTRZ80'].vals * VSQUARED / 133.0
        return P80


#--------------------------------------------
class DP_PABH(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ53', '4OHTRZ54', '4OHTRZ55', '4OHTRZ57']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P53 = data['4OHTRZ53'].vals * VSQUARED / 94.1
        P54 = data['4OHTRZ54'].vals * VSQUARED / 124.4
        P55 = data['4OHTRZ55'].vals * VSQUARED / 126.8
        P57 = data['4OHTRZ57'].vals * VSQUARED / 142.3
        PABH = P53 + P54 + P55 + P57
        return PABH


#--------------------------------------------
class DP_PAFTCONE(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ48', '4OHTRZ49', '4OHTRZ50', '4OHTRZ51', '4OHTRZ52']
    time_step = 0.25625

    def calc(self, data):
        self.fix_4OHTRZ50(data)
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P48 = data['4OHTRZ48'].vals * VSQUARED / 79.5
        P49 = data['4OHTRZ49'].vals * VSQUARED / 34.8
        P50 = data['4OHTRZ50'].vals * VSQUARED / 35.2
        P51 = data['4OHTRZ51'].vals * VSQUARED / 35.4
        P52 = data['4OHTRZ52'].vals * VSQUARED / 34.4
        PAFTCONE = P48 + P49 + P50 + P51 + P52
        return PAFTCONE


#--------------------------------------------
class DP_PAFTCYL(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ66', '4OHTRZ67', '4OHTRZ68']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P66 = data['4OHTRZ66'].vals * VSQUARED / 29.8
        P67 = data['4OHTRZ67'].vals * VSQUARED / 52.0
        P68 = data['4OHTRZ68'].vals * VSQUARED / 29.0
        PAFTCYL = P66 + P67 + P68
        return PAFTCYL


#--------------------------------------------
class DP_PAHP(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ11', '4OHTRZ12', '4OHTRZ13']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P11 = data['4OHTRZ11'].vals * VSQUARED / 39.4
        P12 = data['4OHTRZ12'].vals * VSQUARED / 40.3
        P13 = data['4OHTRZ13'].vals * VSQUARED / 39.7
        PAHP = P11 + P12 + P13
        return PAHP


#--------------------------------------------
class DP_PCONE(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ61', '4OHTRZ62', '4OHTRZ63']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P61 = data['4OHTRZ61'].vals * VSQUARED / 33.7
        P62 = data['4OHTRZ62'].vals * VSQUARED / 36.1
        P63 = data['4OHTRZ63'].vals * VSQUARED / 36.1
        PCONE = P61 + P62 + P63
        return PCONE


#--------------------------------------------
class DP_PFAP(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ01', '4OHTRZ02', '4OHTRZ03', '4OHTRZ04',
                  '4OHTRZ05', '4OHTRZ06', '4OHTRZ07']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P01 = data['4OHTRZ01'].vals * VSQUARED / 110.2
        P02 = data['4OHTRZ02'].vals * VSQUARED / 109.7
        P03 = data['4OHTRZ03'].vals * VSQUARED / 109.4
        P04 = data['4OHTRZ04'].vals * VSQUARED / 175.9
        P05 = data['4OHTRZ05'].vals * VSQUARED / 175.7
        P06 = data['4OHTRZ06'].vals * VSQUARED / 175.6
        P07 = data['4OHTRZ07'].vals * VSQUARED / 135.8
        PFAP = P01 + P02 + P03 + P04 + P05 + P06 + P07
        return PFAP


#--------------------------------------------
class DP_PFWDCONE(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ31', '4OHTRZ32', '4OHTRZ33', '4OHTRZ34',
                  '4OHTRZ35', '4OHTRZ36', '4OHTRZ37', '4OHTRZ38', '4OHTRZ39',
                  '4OHTRZ40']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P31 = data['4OHTRZ31'].vals * VSQUARED / 32.2
        P32 = data['4OHTRZ32'].vals * VSQUARED / 28.6
        P33 = data['4OHTRZ33'].vals * VSQUARED / 36.9
        P34 = data['4OHTRZ34'].vals * VSQUARED / 28.0
        P35 = data['4OHTRZ35'].vals * VSQUARED / 32.2
        P36 = data['4OHTRZ36'].vals * VSQUARED / 44.3
        P37 = data['4OHTRZ37'].vals * VSQUARED / 32.1
        P38 = data['4OHTRZ38'].vals * VSQUARED / 27.8
        P39 = data['4OHTRZ39'].vals * VSQUARED / 36.8
        P40 = data['4OHTRZ40'].vals * VSQUARED / 28.3
        PFWDCONE = (P31 + P32 + P33 + P34 + P35 + P36 + P37 + P38 + P39 +
                    P40)
        return PFWDCONE


#--------------------------------------------
class DP_PFWDCYL(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ58', '4OHTRZ59', '4OHTRZ60']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P58 = data['4OHTRZ58'].vals * VSQUARED / 83.7
        P59 = data['4OHTRZ59'].vals * VSQUARED / 29.7
        P60 = data['4OHTRZ60'].vals * VSQUARED / 30.7
        PFWDCYL = P58 + P59 + P60
        return PFWDCYL


#--------------------------------------------
class DP_PHRMA(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ01', '4OHTRZ02', '4OHTRZ03', '4OHTRZ04',
                  '4OHTRZ05', '4OHTRZ06', '4OHTRZ07', '4OHTRZ08', '4OHTRZ09',
                  '4OHTRZ10', '4OHTRZ11', '4OHTRZ12', '4OHTRZ13', '4OHTRZ14',
                  '4OHTRZ15', '4OHTRZ16', '4OHTRZ17', '4OHTRZ18', '4OHTRZ19',
                  '4OHTRZ20', '4OHTRZ23', '4OHTRZ24']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P01 = data['4OHTRZ01'].vals * VSQUARED / 110.2
        P02 = data['4OHTRZ02'].vals * VSQUARED / 109.7
        P03 = data['4OHTRZ03'].vals * VSQUARED / 109.4
        P04 = data['4OHTRZ04'].vals * VSQUARED / 175.9
        P05 = data['4OHTRZ05'].vals * VSQUARED / 175.7
        P06 = data['4OHTRZ06'].vals * VSQUARED / 175.6
        P07 = data['4OHTRZ07'].vals * VSQUARED / 135.8
        P08 = data['4OHTRZ08'].vals * VSQUARED / 36.1
        P09 = data['4OHTRZ09'].vals * VSQUARED / 32.6
        P10 = data['4OHTRZ10'].vals * VSQUARED / 34.9
        P11 = data['4OHTRZ11'].vals * VSQUARED / 39.4
        P12 = data['4OHTRZ12'].vals * VSQUARED / 40.3
        P13 = data['4OHTRZ13'].vals * VSQUARED / 39.7
        P14 = data['4OHTRZ14'].vals * VSQUARED / 41.2
        P15 = data['4OHTRZ15'].vals * VSQUARED / 40.5
        P16 = data['4OHTRZ16'].vals * VSQUARED / 41.3
        P17 = data['4OHTRZ17'].vals * VSQUARED / 116.0
        P18 = data['4OHTRZ18'].vals * VSQUARED / 115.7
        P19 = data['4OHTRZ19'].vals * VSQUARED / 95.3
        P20 = data['4OHTRZ20'].vals * VSQUARED / 379.0
        P23 = data['4OHTRZ23'].vals * VSQUARED / 386.0
        P24 = data['4OHTRZ24'].vals * VSQUARED / 385.8
        PHRMA = (P01 + P02 + P03 + P04 + P05 + P06 + P07 + P08 + P09 + P10 +
                 P11 + P12 + P13 + P14 + P15 + P16 + P17 + P18 + P19 + P20 +
                 P23 + P24)
        return PHRMA


#--------------------------------------------
class DP_PHRMASTRUTS(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ25', '4OHTRZ26', '4OHTRZ27', '4OHTRZ28',
                  '4OHTRZ29', '4OHTRZ30']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P25 = data['4OHTRZ25'].vals * VSQUARED / 383.0
        P26 = data['4OHTRZ26'].vals * VSQUARED / 383.5
        P27 = data['4OHTRZ27'].vals * VSQUARED / 383.0
        P28 = data['4OHTRZ28'].vals * VSQUARED / 382.3
        P29 = data['4OHTRZ29'].vals * VSQUARED / 384.0
        P30 = data['4OHTRZ30'].vals * VSQUARED / 383.0
        PHRMASTRUTS = P25 + P26 + P27 + P28 + P29 + P30
        return PHRMASTRUTS


#--------------------------------------------
class DP_PIC(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ23', '4OHTRZ24']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P23 = data['4OHTRZ23'].vals * VSQUARED / 386.0
        P24 = data['4OHTRZ24'].vals * VSQUARED / 385.8
        PIC = P23 + P24
        return PIC


#--------------------------------------------
class DP_PMIDCONE(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ41', '4OHTRZ42', '4OHTRZ43', '4OHTRZ44',
                  '4OHTRZ45', '4OHTRZ46', '4OHTRZ47']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P41 = data['4OHTRZ41'].vals * VSQUARED / 61.7
        P42 = data['4OHTRZ42'].vals * VSQUARED / 51.7
        P43 = data['4OHTRZ43'].vals * VSQUARED / 36.8
        P44 = data['4OHTRZ44'].vals * VSQUARED / 36.9
        P45 = data['4OHTRZ45'].vals * VSQUARED / 36.8
        P46 = data['4OHTRZ46'].vals * VSQUARED / 36.5
        P47 = data['4OHTRZ47'].vals * VSQUARED / 52.3
        PMIDCONE = P41 + P42 + P43 + P44 + P45 + P46 + P47
        return PMIDCONE


#--------------------------------------------
class DP_PMNT(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ14', '4OHTRZ15', '4OHTRZ16']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P14 = data['4OHTRZ14'].vals * VSQUARED / 41.2
        P15 = data['4OHTRZ15'].vals * VSQUARED / 40.5
        P16 = data['4OHTRZ16'].vals * VSQUARED / 41.3
        PMNT = P14 + P15 + P16
        return PMNT


#--------------------------------------------
class DP_POBAT(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ25', '4OHTRZ26', '4OHTRZ27', '4OHTRZ28', '4OHTRZ29',
                  '4OHTRZ30', '4OHTRZ31', '4OHTRZ32', '4OHTRZ33', '4OHTRZ34',
                  '4OHTRZ35', '4OHTRZ36', '4OHTRZ37', '4OHTRZ38', '4OHTRZ39',
                  '4OHTRZ40', '4OHTRZ41', '4OHTRZ42', '4OHTRZ43', '4OHTRZ44',
                  '4OHTRZ45', '4OHTRZ46', '4OHTRZ47', '4OHTRZ48', '4OHTRZ49',
                  '4OHTRZ50', '4OHTRZ51', '4OHTRZ52', '4OHTRZ53', '4OHTRZ54',
                  '4OHTRZ55', '4OHTRZ57', '4OHTRZ75', '4OHTRZ76', '4OHTRZ77',
                  '4OHTRZ78', '4OHTRZ79', '4OHTRZ80']
    time_step = 0.25625

    def calc(self, data):
        self.fix_4OHTRZ50(data)
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P75 = data['4OHTRZ75'].vals * VSQUARED / 130.2
        P76 = data['4OHTRZ76'].vals * VSQUARED / 133.4
        P77 = data['4OHTRZ77'].vals * VSQUARED / 131.5
        P78 = data['4OHTRZ78'].vals * VSQUARED / 133.2
        P79 = data['4OHTRZ79'].vals * VSQUARED / 133.1
        P80 = data['4OHTRZ80'].vals * VSQUARED / 133.0
        P25 = data['4OHTRZ25'].vals * VSQUARED / 383.0
        P26 = data['4OHTRZ26'].vals * VSQUARED / 383.5
        P27 = data['4OHTRZ27'].vals * VSQUARED / 383.0
        P28 = data['4OHTRZ28'].vals * VSQUARED / 382.3
        P29 = data['4OHTRZ29'].vals * VSQUARED / 384.0
        P30 = data['4OHTRZ30'].vals * VSQUARED / 383.0

        P31 = data['4OHTRZ31'].vals * VSQUARED / 32.2
        P32 = data['4OHTRZ32'].vals * VSQUARED / 28.6
        P33 = data['4OHTRZ33'].vals * VSQUARED / 36.9
        P34 = data['4OHTRZ34'].vals * VSQUARED / 28.0
        P35 = data['4OHTRZ35'].vals * VSQUARED / 32.2
        P36 = data['4OHTRZ36'].vals * VSQUARED / 44.3
        P37 = data['4OHTRZ37'].vals * VSQUARED / 32.1
        P38 = data['4OHTRZ38'].vals * VSQUARED / 27.8
        P39 = data['4OHTRZ39'].vals * VSQUARED / 36.8
        P40 = data['4OHTRZ40'].vals * VSQUARED / 28.3
        P41 = data['4OHTRZ41'].vals * VSQUARED / 61.7
        P42 = data['4OHTRZ42'].vals * VSQUARED / 51.7
        P43 = data['4OHTRZ43'].vals * VSQUARED / 36.8
        P44 = data['4OHTRZ44'].vals * VSQUARED / 36.9
        P45 = data['4OHTRZ45'].vals * VSQUARED / 36.8
        P46 = data['4OHTRZ46'].vals * VSQUARED / 36.5
        P47 = data['4OHTRZ47'].vals * VSQUARED / 52.3
        P48 = data['4OHTRZ48'].vals * VSQUARED / 79.5
        P49 = data['4OHTRZ49'].vals * VSQUARED / 34.8
        P50 = data['4OHTRZ50'].vals * VSQUARED / 35.2
        P51 = data['4OHTRZ51'].vals * VSQUARED / 35.4
        P52 = data['4OHTRZ52'].vals * VSQUARED / 34.4
        P53 = data['4OHTRZ53'].vals * VSQUARED / 94.1
        P54 = data['4OHTRZ54'].vals * VSQUARED / 124.4
        P55 = data['4OHTRZ55'].vals * VSQUARED / 126.8
        P57 = data['4OHTRZ57'].vals * VSQUARED / 142.3

        PSTRUTS = (P75 + P76 + P77 + P78 + P79 + P80 + P25 + P26 + P27 + P28 +
                   P29 + P30)
        POBACONE = (P31 + P32 + P33 + P34 + P35 + P36 + P37 + P38 + P39 + P40 +
                    P41 + P42 + P43 + P44 + P45 + P46 + P47 + P48 + P49 + P50 +
                    P51 + P52 + P53 + P54 + P55 + P57)
        POBAT = PSTRUTS + POBACONE
        return POBAT


#--------------------------------------------
class DP_POC(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ17', '4OHTRZ18', '4OHTRZ19']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P17 = data['4OHTRZ17'].vals * VSQUARED / 116.0
        P18 = data['4OHTRZ18'].vals * VSQUARED / 115.7
        P19 = data['4OHTRZ19'].vals * VSQUARED / 95.3
        POC = P17 + P18 + P19
        return POC


#--------------------------------------------
class DP_PPL10(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ08', '4OHTRZ09', '4OHTRZ10']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P08 = data['4OHTRZ08'].vals * VSQUARED / 36.1
        P09 = data['4OHTRZ09'].vals * VSQUARED / 32.6
        P10 = data['4OHTRZ10'].vals * VSQUARED / 34.9
        PPL10 = P08 + P09 + P10
        return PPL10


#--------------------------------------------
class DP_PRADVNT(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ64', '4OHTRZ65', '4OHTRZ69']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P64 = data['4OHTRZ64'].vals * VSQUARED / 44.1
        P65 = data['4OHTRZ65'].vals * VSQUARED / 37.5
        P69 = data['4OHTRZ69'].vals * VSQUARED / 37.5
        PRADVNT = P64 + P65 + P69
        return PRADVNT


#--------------------------------------------
class DP_PSCSTRUTS(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ75', '4OHTRZ76', '4OHTRZ77',
                  '4OHTRZ78', '4OHTRZ79', '4OHTRZ80']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P75 = data['4OHTRZ75'].vals * VSQUARED / 130.2
        P76 = data['4OHTRZ76'].vals * VSQUARED / 133.4
        P77 = data['4OHTRZ77'].vals * VSQUARED / 131.5
        P78 = data['4OHTRZ78'].vals * VSQUARED / 133.2
        P79 = data['4OHTRZ79'].vals * VSQUARED / 133.1
        P80 = data['4OHTRZ80'].vals * VSQUARED / 133.0
        PSCSTRUTS = P75 + P76 + P77 + P78 + P79 + P80
        return PSCSTRUTS


#--------------------------------------------
class DP_PTFTE(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ58', '4OHTRZ59', '4OHTRZ60', '4OHTRZ61',
                  '4OHTRZ62', '4OHTRZ63', '4OHTRZ64', '4OHTRZ65', '4OHTRZ66',
                  '4OHTRZ67', '4OHTRZ68', '4OHTRZ69']
    time_step = 0.25625

    def calc(self, data):
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P58 = data['4OHTRZ58'].vals * VSQUARED / 83.7
        P59 = data['4OHTRZ59'].vals * VSQUARED / 29.7
        P60 = data['4OHTRZ60'].vals * VSQUARED / 30.7
        P61 = data['4OHTRZ61'].vals * VSQUARED / 33.7
        P62 = data['4OHTRZ62'].vals * VSQUARED / 36.1
        P63 = data['4OHTRZ63'].vals * VSQUARED / 36.1
        P64 = data['4OHTRZ64'].vals * VSQUARED / 44.1
        P65 = data['4OHTRZ65'].vals * VSQUARED / 37.5
        P66 = data['4OHTRZ66'].vals * VSQUARED / 29.8
        P67 = data['4OHTRZ67'].vals * VSQUARED / 52.0
        P68 = data['4OHTRZ68'].vals * VSQUARED / 29.0
        P69 = data['4OHTRZ69'].vals * VSQUARED / 37.5

        PTFTE = (P58 + P59 + P60 + P61 + P62 + P63 + P64 + P65 + P66 + P67 +
                 P68 + P69)
        return PTFTE


#--------------------------------------------
class DP_PTOTAL(DerivedParameterThermal):
    rootparams = ['ELBV', '4OHTRZ01', '4OHTRZ02', '4OHTRZ03', '4OHTRZ04',
                  '4OHTRZ05', '4OHTRZ06', '4OHTRZ07', '4OHTRZ08', '4OHTRZ09',
                  '4OHTRZ10', '4OHTRZ11', '4OHTRZ12', '4OHTRZ13', '4OHTRZ14',
                  '4OHTRZ15', '4OHTRZ16', '4OHTRZ17', '4OHTRZ18', '4OHTRZ19',
                  '4OHTRZ20', '4OHTRZ23', '4OHTRZ24', '4OHTRZ25', '4OHTRZ26',
                  '4OHTRZ27', '4OHTRZ28', '4OHTRZ29', '4OHTRZ30', '4OHTRZ31',
                  '4OHTRZ32', '4OHTRZ33', '4OHTRZ34', '4OHTRZ35', '4OHTRZ36',
                  '4OHTRZ37', '4OHTRZ38', '4OHTRZ39', '4OHTRZ40', '4OHTRZ41',
                  '4OHTRZ42', '4OHTRZ43', '4OHTRZ44', '4OHTRZ45', '4OHTRZ46',
                  '4OHTRZ47', '4OHTRZ48', '4OHTRZ49', '4OHTRZ50', '4OHTRZ51',
                  '4OHTRZ52', '4OHTRZ53', '4OHTRZ54', '4OHTRZ55', '4OHTRZ57',
                  '4OHTRZ58', '4OHTRZ59', '4OHTRZ60', '4OHTRZ61', '4OHTRZ62',
                  '4OHTRZ63', '4OHTRZ64', '4OHTRZ65', '4OHTRZ66', '4OHTRZ67',
                  '4OHTRZ68', '4OHTRZ69', '4OHTRZ75', '4OHTRZ76', '4OHTRZ77',
                  '4OHTRZ78', '4OHTRZ79', '4OHTRZ80']
    time_step = 0.25625

    def calc(self, data):
        self.fix_4OHTRZ50(data)
        VSQUARED = data['ELBV'].vals * data['ELBV'].vals
        P01 = data['4OHTRZ01'].vals * VSQUARED / 110.2
        P02 = data['4OHTRZ02'].vals * VSQUARED / 109.7
        P03 = data['4OHTRZ03'].vals * VSQUARED / 109.4
        P04 = data['4OHTRZ04'].vals * VSQUARED / 175.9
        P05 = data['4OHTRZ05'].vals * VSQUARED / 175.7
        P06 = data['4OHTRZ06'].vals * VSQUARED / 175.6
        P07 = data['4OHTRZ07'].vals * VSQUARED / 135.8
        P08 = data['4OHTRZ08'].vals * VSQUARED / 36.1
        P09 = data['4OHTRZ09'].vals * VSQUARED / 32.6
        P10 = data['4OHTRZ10'].vals * VSQUARED / 34.9
        P11 = data['4OHTRZ11'].vals * VSQUARED / 39.4
        P12 = data['4OHTRZ12'].vals * VSQUARED / 40.3
        P13 = data['4OHTRZ13'].vals * VSQUARED / 39.7
        P14 = data['4OHTRZ14'].vals * VSQUARED / 41.2
        P15 = data['4OHTRZ15'].vals * VSQUARED / 40.5
        P16 = data['4OHTRZ16'].vals * VSQUARED / 41.3
        P17 = data['4OHTRZ17'].vals * VSQUARED / 116.0
        P18 = data['4OHTRZ18'].vals * VSQUARED / 115.7
        P19 = data['4OHTRZ19'].vals * VSQUARED / 95.3
        P20 = data['4OHTRZ20'].vals * VSQUARED / 379.0
        P23 = data['4OHTRZ23'].vals * VSQUARED / 386.0
        P24 = data['4OHTRZ24'].vals * VSQUARED / 385.8

        P31 = data['4OHTRZ31'].vals * VSQUARED / 32.2
        P32 = data['4OHTRZ32'].vals * VSQUARED / 28.6
        P33 = data['4OHTRZ33'].vals * VSQUARED / 36.9
        P34 = data['4OHTRZ34'].vals * VSQUARED / 28.0
        P35 = data['4OHTRZ35'].vals * VSQUARED / 32.2
        P36 = data['4OHTRZ36'].vals * VSQUARED / 44.3
        P37 = data['4OHTRZ37'].vals * VSQUARED / 32.1
        P38 = data['4OHTRZ38'].vals * VSQUARED / 27.8
        P39 = data['4OHTRZ39'].vals * VSQUARED / 36.8
        P40 = data['4OHTRZ40'].vals * VSQUARED / 28.3
        P41 = data['4OHTRZ41'].vals * VSQUARED / 61.7
        P42 = data['4OHTRZ42'].vals * VSQUARED / 51.7
        P43 = data['4OHTRZ43'].vals * VSQUARED / 36.8
        P44 = data['4OHTRZ44'].vals * VSQUARED / 36.9
        P45 = data['4OHTRZ45'].vals * VSQUARED / 36.8
        P46 = data['4OHTRZ46'].vals * VSQUARED / 36.5
        P47 = data['4OHTRZ47'].vals * VSQUARED / 52.3
        P48 = data['4OHTRZ48'].vals * VSQUARED / 79.5
        P49 = data['4OHTRZ49'].vals * VSQUARED / 34.8
        P50 = data['4OHTRZ50'].vals * VSQUARED / 35.2
        P51 = data['4OHTRZ51'].vals * VSQUARED / 35.4
        P52 = data['4OHTRZ52'].vals * VSQUARED / 34.4
        P53 = data['4OHTRZ53'].vals * VSQUARED / 94.1
        P54 = data['4OHTRZ54'].vals * VSQUARED / 124.4
        P55 = data['4OHTRZ55'].vals * VSQUARED / 126.8
        P57 = data['4OHTRZ57'].vals * VSQUARED / 142.3

        P58 = data['4OHTRZ58'].vals * VSQUARED / 83.7
        P59 = data['4OHTRZ59'].vals * VSQUARED / 29.7
        P60 = data['4OHTRZ60'].vals * VSQUARED / 30.7
        P61 = data['4OHTRZ61'].vals * VSQUARED / 33.7
        P62 = data['4OHTRZ62'].vals * VSQUARED / 36.1
        P63 = data['4OHTRZ63'].vals * VSQUARED / 36.1
        P64 = data['4OHTRZ64'].vals * VSQUARED / 44.1
        P65 = data['4OHTRZ65'].vals * VSQUARED / 37.5
        P66 = data['4OHTRZ66'].vals * VSQUARED / 29.8
        P67 = data['4OHTRZ67'].vals * VSQUARED / 52.0
        P68 = data['4OHTRZ68'].vals * VSQUARED / 29.0
        P69 = data['4OHTRZ69'].vals * VSQUARED / 37.5

        P75 = data['4OHTRZ75'].vals * VSQUARED / 130.2
        P76 = data['4OHTRZ76'].vals * VSQUARED / 133.4
        P77 = data['4OHTRZ77'].vals * VSQUARED / 131.5
        P78 = data['4OHTRZ78'].vals * VSQUARED / 133.2
        P79 = data['4OHTRZ79'].vals * VSQUARED / 133.1
        P80 = data['4OHTRZ80'].vals * VSQUARED / 133.0
        P25 = data['4OHTRZ25'].vals * VSQUARED / 383.0
        P26 = data['4OHTRZ26'].vals * VSQUARED / 383.5
        P27 = data['4OHTRZ27'].vals * VSQUARED / 383.0
        P28 = data['4OHTRZ28'].vals * VSQUARED / 382.3
        P29 = data['4OHTRZ29'].vals * VSQUARED / 384.0
        P30 = data['4OHTRZ30'].vals * VSQUARED / 383.0

        PHRMA = (P01 + P02 + P03 + P04 + P05 + P06 + P07 + P08 + P09 + P10 +
                 P11 + P12 + P13 + P14 + P15 + P16 + P17 + P18 + P19 + P20 +
                 P23 + P24)
        POBACONE = (P31 + P32 + P33 + P34 + P35 + P36 + P37 + P38 + P39 + P40 +
                    P41 + P42 + P43 + P44 + P45 + P46 + P47 + P48 + P49 + P50 +
                    P51 + P52 + P53 + P54 + P55 + P57)
        PTFTE = (P58 + P59 + P60 + P61 + P62 + P63 + P64 + P65 + P66 + P67 +
                 P68 + P69)
        PSTRUTS = (P75 + P76 + P77 + P78 + P79 + P80 + P25 + P26 + P27 + P28 +
                   P29 + P30)

        POBAT = PSTRUTS + POBACONE
        PTOTAL = PHRMA + POBAT + PTFTE
        return PTOTAL


#--------------------------------------------
class DP_SUNANGLE(DerivedParameterThermal):
    rootparams = ['AOSARES1']
    time_step = 0.25625

    def calc(self, data):
        SUNANGLE = (90 - data['AOSARES1'].vals)
        return SUNANGLE


#--------------------------------------------
class DP_TABMAX(DerivedParameterThermal):
    rootparams = ['OOBTHR47', 'OOBTHR42', 'OOBTHR43']
    time_step = 32.8

    def calc(self, data):
        TABMAX = np.max([data['OOBTHR42'].vals, data['OOBTHR43'].vals,
                         data['OOBTHR47'].vals], axis=0)
        return TABMAX


#--------------------------------------------
class DP_TABMIN(DerivedParameterThermal):
    rootparams = ['OOBTHR47', 'OOBTHR42', 'OOBTHR43']
    time_step = 32.8

    def calc(self, data):
        TABMIN = np.min([data['OOBTHR42'].vals, data['OOBTHR43'].vals,
                         data['OOBTHR47'].vals], axis=0)
        return TABMIN


#--------------------------------------------
class DP_TELAB_AVE(DerivedParameterThermal):
    rootparams = ['OOBTHR47', 'OOBTHR42', 'OOBTHR43']
    time_step = 32.8

    def calc(self, data):
        TELAB_AVE = (data['OOBTHR42'].vals + data['OOBTHR43'].vals +
                     data['OOBTHR47'].vals) / 3
        return TELAB_AVE


#--------------------------------------------
class DP_TELHS_AVE(DerivedParameterThermal):
    rootparams = ['OOBTHR02', 'OOBTHR03', 'OOBTHR06', 'OOBTHR07', 'OOBTHR04',
                  'OOBTHR05']
    time_step = 32.8

    def calc(self, data):
        TELHS_AVE = (data['OOBTHR02'].vals + data['OOBTHR03'].vals +
                     data['OOBTHR04'].vals + data['OOBTHR05'].vals +
                     data['OOBTHR06'].vals + data['OOBTHR07'].vals) / 6
        return TELHS_AVE


#--------------------------------------------
class DP_TELSS_AVE(DerivedParameterThermal):
    rootparams = ['OOBTHR51', 'OOBTHR50', 'OOBTHR53', 'OOBTHR52', 'OOBTHR54',
                  'OOBTHR49']
    time_step = 32.8

    def calc(self, data):
        TELSS_AVE = (data['OOBTHR49'].vals + data['OOBTHR50'].vals +
                     data['OOBTHR51'].vals + data['OOBTHR52'].vals +
                     data['OOBTHR53'].vals + data['OOBTHR54'].vals) / 6
        return TELSS_AVE


#--------------------------------------------
class DP_THSMAX(DerivedParameterThermal):
    rootparams = ['OOBTHR02', 'OOBTHR03', 'OOBTHR06', 'OOBTHR07', 'OOBTHR04',
                  'OOBTHR05']
    time_step = 32.8

    def calc(self, data):
        THSMAX = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            THSMAX = np.max([THSMAX, data[names].vals], axis=0)
        return THSMAX


#--------------------------------------------
class DP_THSMIN(DerivedParameterThermal):
    rootparams = ['OOBTHR02', 'OOBTHR03', 'OOBTHR06', 'OOBTHR07', 'OOBTHR04',
                  'OOBTHR05']
    time_step = 32.8

    def calc(self, data):
        THSMIN = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            THSMIN = np.min([THSMIN, data[names].vals], axis=0)
        return THSMIN


#--------------------------------------------
class DP_TILT_AXIAL(DerivedParameterThermal):
    rootparams = ['OOBAGRD3']
    time_step = 32.8

    def calc(self, data):
        TILT_AXIAL = np.abs(data['OOBAGRD3'].vals) * 0.1084
        return TILT_AXIAL


#--------------------------------------------
class DP_TILT_BULK(DerivedParameterThermal):
    rootparams = ['OHRTHR43', 'OHRTHR42']
    time_step = 32.8

    def calc(self, data):
        TILT_BULK = np.abs((data['OHRTHR42'].vals + data['OHRTHR43'].vals)
                           / 2.0 - 70.0) * 0.0704
        return TILT_BULK


#--------------------------------------------
class DP_TILT_DIAM(DerivedParameterThermal):
    rootparams = ['OOBAGRD6']
    time_step = 32.8

    def calc(self, data):
        TILT_DIAM = np.abs(1.0 * data['OOBAGRD6'].vals) * 0.3032
        return TILT_DIAM


#--------------------------------------------
class DP_TILT_MAX(DerivedParameterThermal):
    rootparams = ['OOBAGRD6', 'OOBAGRD3', 'OHRTHR43', 'OHRTHR42']
    time_step = 32.8

    def calc(self, data):
        TILT_DIAM = np.abs(1.0 * data['OOBAGRD6'].vals) * 0.3032
        TILT_BULK = np.abs((data['OHRTHR42'].vals + data['OHRTHR43'].vals)
                           / 2.0 - 70.0) * 0.0704
        TILT_AXIAL = np.abs(data['OOBAGRD3'].vals) * 0.1084
        TILT_MAX = (TILT_BULK + TILT_AXIAL + TILT_DIAM)
        return TILT_MAX


#--------------------------------------------
class DP_TILT_RSS(DerivedParameterThermal):
    rootparams = ['OOBAGRD6', 'OOBAGRD3', 'OHRTHR43', 'OHRTHR42']
    time_step = 32.8

    def calc(self, data):
        TILT_DIAM = np.abs(1.0 * data['OOBAGRD6'].vals) * 0.3032
        TILT_BULK = np.abs((data['OHRTHR42'].vals + data['OHRTHR43'].vals)
                           / 2.0 - 70.0) * 0.0704
        TILT_AXIAL = np.abs(data['OOBAGRD3'].vals) * 0.1084
        TILT_RSS = np.sqrt(TILT_BULK ** 2 + TILT_AXIAL ** 2 + TILT_DIAM ** 2)
        return TILT_RSS


#--------------------------------------------
class DP_TSSMAX(DerivedParameterThermal):
    rootparams = ['OOBTHR51', 'OOBTHR50', 'OOBTHR53', 'OOBTHR52', 'OOBTHR54',
                  'OOBTHR49']
    time_step = 32.8

    def calc(self, data):
        TSSMAX = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            TSSMAX = np.max([TSSMAX, data[names].vals], axis=0)
        return TSSMAX


#--------------------------------------------
class DP_TSSMIN(DerivedParameterThermal):
    rootparams = ['OOBTHR51', 'OOBTHR50', 'OOBTHR53', 'OOBTHR52', 'OOBTHR54',
                  'OOBTHR49']
    time_step = 32.8

    def calc(self, data):
        TSSMIN = data[self.rootparams[0]].vals
        for names in self.rootparams[1:]:
            TSSMIN = np.min([TSSMIN, data[names].vals], axis=0)
        return TSSMIN


#--------------------------------------------
class DP_HADG(DerivedParameterThermal):
    rootparams = ['OHRMGRD3', 'OHRMGRD6']
    time_step = 32.8

    def calc(self, data):
        HADG = np.max((data['OHRMGRD3'].vals, data['OHRMGRD6'].vals), axis=0)

        return HADG


#--------------------------------------------
class DP_ABH_DUTYCYCLE(DerivedParameterThermal):
    rootparams = ['4OHTRZ53', '4OHTRZ54', '4OHTRZ55', '4OHTRZ57']
    time_step = 32.8

    def calc(self, data):
        RTOTAL = 1. / (1 / 94.1 + 1 / 124.3 + 1 / 126.8 + 1 / 142.3)
        DC1 = np.abs(data['4OHTRZ53'].vals) / 94.1 + np.abs(data['4OHTRZ54'].vals) / 124.3
        DC2 = np.abs(data['4OHTRZ55'].vals) / 126.8 + np.abs(data['4OHTRZ57'].vals) / 142.3
        ABH_DUTYCYCLE = RTOTAL * (DC1 + DC2)

        return ABH_DUTYCYCLE
