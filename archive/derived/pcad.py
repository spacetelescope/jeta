# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Derived parameter MSIDs related to PCAD subsystem.

Author: A. Arvai

Revision History::

     Jan 2012       Initial version
   1 Mar 2012       Modified all ephemeris-based parameters to use predictive
                    ephemeris
  26 Mar 2012       Re-defined DP_ROLL_FSS and DP_PITCH_FSS to improve accuracy
"""

import numpy as np
from numpy import sin, cos, tan, arctan2, sqrt, degrees, radians
from . import base


class DerivedParameterPcad(base.DerivedParameter):
    content_root = 'pcad'


#--------------------------------------------
class DP_CSS1_NPM_SUN(DerivedParameterPcad):
    """Coarse Sun Sensor Counts 1 filtered for NPM and SA Illuminated

    Defined as CSS-1 current converted back into counts
    (AOCSSI1 * 4095 / 5.49549) when in NPM (AOPCADMD==1) and SA is illuminated
    (AOSAILLM==1).  Otherwise, "Bads" flag is set equal to one.

    """
    rootparams = ['aocssi1', 'aopcadmd', 'aosaillm']
    time_step = 1.025
    max_gap = 10.0

    def calc(self, data):
        npm_sun = ((data['aopcadmd'].vals == 'NPNT') &
                   (data['aosaillm'].vals == 'ILLM'))
        data.bads = data.bads | ~npm_sun
        css1_npm_sun = data['aocssi1'].vals * 4095 / 5.49549
        return css1_npm_sun


#--------------------------------------------
class DP_CSS2_NPM_SUN(DerivedParameterPcad):
    """Coarse Sun Sensor Counts 2 filtered for NPM and SA Illuminated

    Defined as CSS-2 current converted back into counts
    (AOCSSI2 * 4095 / 5.49549) when in NPM (AOPCADMD==1) and SA is illuminated
    (AOSAILLM==1).  Otherwise, "Bads" flag is set equal to one.

    """
    rootparams = ['aocssi2', 'aopcadmd', 'aosaillm']
    time_step = 1.025
    max_gap = 10.0

    def calc(self, data):
        npm_sun = ((data['aopcadmd'].vals == 'NPNT') &
                   (data['aosaillm'].vals == 'ILLM'))
        data.bads = data.bads | ~npm_sun
        css2_npm_sun = data['aocssi2'].vals * 4095 / 5.49549
        return css2_npm_sun


#--------------------------------------------
class DP_CSS3_NPM_SUN(DerivedParameterPcad):
    """Coarse Sun Sensor Counts 3 filtered for NPM and SA Illuminated

    Defined as CSS-3 current converted back into counts
    (AOCSSI3 * 4095 / 5.49549) when in NPM (AOPCADMD==1) and SA is illuminated
    (AOSAILLM==1).  Otherwise, "Bads" flag is set equal to one.

    """
    rootparams = ['aocssi3', 'aopcadmd', 'aosaillm']
    time_step = 1.025
    max_gap = 10.0

    def calc(self, data):
        npm_sun = ((data['aopcadmd'].vals == 'NPNT') &
                   (data['aosaillm'].vals == 'ILLM'))
        data.bads = data.bads | ~npm_sun
        css3_npm_sun = data['aocssi3'].vals * 4095 / 5.49549
        return css3_npm_sun


#--------------------------------------------
class DP_CSS4_NPM_SUN(DerivedParameterPcad):
    """Coarse Sun Sensor Counts 4 filtered for NPM and SA Illuminated

    Defined as CSS-4 current converted back into counts
    (AOCSSI4 * 4095 / 5.49549) when in NPM (AOPCADMD==1) and SA is illuminated
    (AOSAILLM==1).  Otherwise, "Bads" flag is set equal to one.

    """
    rootparams = ['aocssi4', 'aopcadmd', 'aosaillm']
    time_step = 1.025
    max_gap = 10.0

    def calc(self, data):
        npm_sun = ((data['aopcadmd'].vals == 'NPNT') &
                   (data['aosaillm'].vals == 'ILLM'))
        data.bads = data.bads | ~npm_sun
        css4_npm_sun = data['aocssi4'].vals * 4095 / 5.49549
        return css4_npm_sun


#--------------------------------------------
class DP_FSS_CSS_ANGLE_DIFF(DerivedParameterPcad):
    """Angle between FSS and CSS Sun Vectors [Deg]

    Defined as the angle between the FSS and CSS sun vectors

    Calculated by rotating the CSS sun vector from the SA-1 frame to ACA frame
    then computing the angular difference using the dot product and ARCCOS.
    "Bads" flag is set equal to one when not in the FSS FOV.

    """
    rootparams = ['aosunsa1', 'aosunsa2', 'aosunsa3',
                  'aosunac1', 'aosunac2', 'aosunac3',
                  'aosares1', 'aosares2', 'aosunprs']
    time_step = 1.025
    max_gap = 18.0
    dtype = np.float32

    def calc(self, data):
        in_fss_fov = (data['aosunprs'].vals == 'SUN ')
        data.bads |= ~in_fss_fov
        sa_ang_avg = (data['aosares1'].vals + data['aosares2'].vals) / 2
        sinang = sin(radians(sa_ang_avg))
        cosang = cos(radians(sa_ang_avg))
        fss_aca = np.array([data['aosunac1'].vals,
                            data['aosunac2'].vals,
                            data['aosunac3'].vals])
        #Rotate CSS sun vector from SA to ACA frame
        css_aca = np.array([sinang * data['aosunsa1'].vals -
                            cosang * data['aosunsa3'].vals,
                            data['aosunsa2'].vals * 1.0,
                            cosang * data['aosunsa1'].vals +
                            sinang * data['aosunsa3'].vals])
        #Normalize the vectors (again)
        magnitude = sqrt((fss_aca * fss_aca).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        fss_aca = fss_aca / magnitude

        magnitude = sqrt((css_aca * css_aca).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        css_aca = css_aca / magnitude

        #Compute the angle between the vectors
        dot_prod = (css_aca * fss_aca).sum(axis=0)
        fss_css_angle_diff = degrees(np.abs(arccos_clip(dot_prod)))
        return fss_css_angle_diff


#--------------------------------------------
class DP_MAN_ANG(DerivedParameterPcad):
    """Maneuver Angle (Total)  [deg]

    Defined as the  angle between the estimated quaternion and the target
    quaternion during a maneuver.

    Computed using the fourth component of the delta quaternion between
    AOTARQT<N> and AOATTQT<N> when a maneuver is in progress
    (AOMANEND = NEND), otherwise equal to zero.

    """
    rootparams = ['aoattqt1', 'aoattqt2', 'aoattqt3', 'aoattqt4',
                  'aotarqt1', 'aotarqt2', 'aotarqt3', 'aomanend']
    time_step = 1.025
    dtype = np.float32

    def calc(self, data):
        qt4_sqr = 1.0 - (data['aotarqt1'].vals ** 2 +
                         data['aotarqt2'].vals ** 2 +
                         data['aotarqt3'].vals ** 2)
        aotarqt4 = sqrt(np.clip(qt4_sqr, 0, 1))
        est_quat_inv = np.array([-1 * data['aoattqt1'].vals,
                                 -1 * data['aoattqt2'].vals,
                                 -1 * data['aoattqt3'].vals,
                                      data['aoattqt4'].vals])
        tar_quat = np.array([data['aotarqt1'].vals,
                             data['aotarqt2'].vals,
                             data['aotarqt3'].vals,
                                   aotarqt4])
        delta_quat = qmult(est_quat_inv, tar_quat)
        # Normalize delta_quat due to roundoff errors.
        magnitude = sqrt((delta_quat * delta_quat).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        delta_quat3 = np.abs(delta_quat[3, :] / magnitude)
        man_ang = 2.0 * degrees(arccos_clip(delta_quat3))

        man = (data['aomanend'].vals == 'NEND')
        man_ang[~man] = 0
        return man_ang


#--------------------------------------------
class DP_ONE_SHOT(DerivedParameterPcad):
    """One Shot [arcsec]

    Defined as the RSS of AOATTER2 and AOATTER3 while in NPM and zero for all
    other PCAD modes.

    """
    rootparams = ['aoatter2', 'aoatter3', 'aopcadmd']
    time_step = 1.025
    max_gap = 4.0
    dtype = np.float32

    def calc(self, data):
        one_shot = degrees(sqrt(data['aoatter2'].vals ** 2 +
                                data['aoatter3'].vals ** 2)) * 3600
        npm = (data['aopcadmd'].vals == 'NPNT')
        one_shot[~npm] = 0.0
        return one_shot


#--------------------------------------------
class DP_PITCH(DerivedParameterPcad):
    """Sun Pitch Angle from Predictive Ephemeris in ACA Frame [deg]

    Defined as the angle between the sun vector and ACA X-axis.

    Calculated using arccos of the sun vector x component in the body frame
    where the sun vector is from predictive ephemeris
    [SOLAREPHEM0 and ORBITEPHEM0] and the estimated attitude from the OBC's
    estimated quaternion [AOATTQT<n>].

    """
    rootparams = ['orbitephem0_x', 'orbitephem0_y', 'orbitephem0_z',
                  'solarephem0_x', 'solarephem0_y', 'solarephem0_z',
                  'aoattqt1', 'aoattqt2', 'aoattqt3',
                  'aoattqt4']
    time_step = 1.025
    max_gap = 4.0
    max_gaps = {msid: 602.0 for msid in rootparams if 'ephem' in msid}
    dtype = np.float32

    def calc(self, data):
        sun_vec_b = sun_vector_body(data)
        pitch = degrees(arccos_clip(sun_vec_b[0, :]))
        return pitch


#--------------------------------------------
class DP_PITCH_CSS(DerivedParameterPcad):
    """Sun Pitch Angle from CSS Data in ACA Frame [Deg]

    Defined as the angle between the sun vector and ACA X-axis.

    Calculated by rotating the CSS sun vector from the SA-1 frame to ACA frame
    based on the solar array angles AOSARES1 and AOSARES2.

    """
    rootparams = ['aosares1', 'aosares2', 'aosunsa1', 'aosunsa2', 'aosunsa3']
    time_step = 4.1
    max_gap = 18.0
    dtype = np.float32

    def calc(self, data):
        sa_ang_avg = (1.0 * data['aosares1'].vals +
                      1.0 * data['aosares2'].vals) / 2
        sinang = sin(radians(sa_ang_avg))
        cosang = cos(radians(sa_ang_avg))
        #Rotate CSS sun vector from SA to ACA frame
        css_aca = np.array([sinang * data['aosunsa1'].vals -
                            cosang * data['aosunsa3'].vals,
                            data['aosunsa2'].vals * 1.0,
                            cosang * data['aosunsa1'].vals +
                            sinang * data['aosunsa3'].vals])
        #Normalize sun vec (again) and compute pitch
        magnitude = sqrt((css_aca * css_aca).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        sun_vec_norm = css_aca / magnitude
        pitch_css = degrees(arccos_clip(sun_vec_norm[0]))
        return pitch_css


#--------------------------------------------
class DP_PITCH_CSS_SA(DerivedParameterPcad):
    """Sun Pitch Angle from CSS Data in SA Frame [Deg]

    Defined as the rotation about the SA-1 Y-axis required to align the sun
    vector with the SA-1 Y-Z plane.

    Calculated as 90.0 - ARCCOS(AOSUNSA1).

    """
    rootparams = ['aosunsa1']
    time_step = 8.2
    max_gap = 18.0
    dtype = np.float32

    def calc(self, data):
        pitch_css_sa = 90.0 - degrees(arccos_clip(data['aosunsa1'].vals))
        return pitch_css_sa


#--------------------------------------------
class DP_PITCH_FSS(DerivedParameterPcad):
    """Sun Pitch Angle from FSS Data in ACA Frame [Deg]

    Defined as the angle between the sun vector and ACA X-axis.

    When in FSS FOV per AOSUNPRS:
    Calculated using the FSS alpha and beta angles to compute the sun vector
    in the FSS frame.  The sun vector is then rotated into the ACA frame
    using the rotation matrix (an OBC k-constant).  Pitch is computed using the
    arccos function.

    When NOT in FSS FOV per AOSUNPRS:
    <data>.bads = 1
    """
    rootparams = ['aoalpang', 'aobetang', 'aosunprs']
    time_step = 1.025
    max_gap = 10.0
    dtype = np.float32

    def calc(self, data):
        in_fss_fov = (data['aosunprs'].vals == 'SUN ')
        data.bads = data.bads | ~in_fss_fov
        # rotation matrix from FSS to ACA frame
        A_AF = np.array([[9.999990450374580e-01,
                           0.0,
                          -1.382000062241829e-03],
                         [-5.327615067743422e-07,
                           9.999999256947376e-01,
                          -3.854999811959735e-04],
                         [1.381999959551952e-03,
                           3.855003493343671e-04,
                           9.999989707322665e-01]])
        # FSS's sun vector in FSS frame
        alpha = radians(data['aoalpang'].vals)
        beta = radians(data['aobetang'].vals)
        sun_fss = np.array([tan(beta), tan(alpha), -np.ones(len(alpha))])
        sun_aca = A_AF.dot(sun_fss)
        magnitude = sqrt((sun_aca * sun_aca).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        sun_vec_norm = sun_aca / magnitude
        pitch_fss = degrees(arccos_clip(sun_vec_norm[0]))
        return pitch_fss


#--------------------------------------------
class DP_ROLL(DerivedParameterPcad):
    """Off-Nominal Roll Angle in ACA Frame [Deg]

    Defined as the rotation about the ACA X-axis required to align the sun
    vector with the ACA X/Z plane.

    Calculated using the four-quadrant arctan of the sun vector y and z
    components in the ACA frame where the sun vector is from predictive
    ephemeris [SOLAREPHEM0 and ORBITEPHEM0] and the estimated attitude from
    the OBC's estimated quaternion [AOATTQT<n>].

    http://occweb.cfa.harvard.edu/twiki/pub/Aspect/WebHome/ROLLDEV3.pdf
    """
    rootparams = ['orbitephem0_x', 'orbitephem0_y', 'orbitephem0_z',
                  'solarephem0_x', 'solarephem0_y', 'solarephem0_z',
                  'aoattqt1', 'aoattqt2', 'aoattqt3', 'aoattqt4']
    time_step = 1.025
    max_gap = 4.0
    max_gaps = {msid: 602.0 for msid in rootparams if 'ephem' in msid}
    dtype = np.float32

    def calc(self, data):
        sun_vec_b = sun_vector_body(data)
        roll = degrees(arctan2(-sun_vec_b[1, :], -sun_vec_b[2, :]))
        return roll


#--------------------------------------------
class DP_ROLL_CSS(DerivedParameterPcad):
    """Off-Nominal Roll Angle from CSS Data in ACA Frame [Deg]

    Defined as the rotation about the ACA X-axis required to align the sun
    vector with the ACA X/Z plane.

    Calculated by rotating the CSS sun vector from the SA-1 frame to ACA frame
    based on the solar array angles AOSARES1 and AOSARES2.

    """
    rootparams = ['aosares1', 'aosares2', 'aosunsa1', 'aosunsa2', 'aosunsa3']
    time_step = 4.1
    max_gap = 18.0
    dtype = np.float32

    def calc(self, data):
        sa_ang_avg = (data['aosares1'].vals + data['aosares2'].vals) / 2
        sinang = sin(radians(sa_ang_avg))
        cosang = cos(radians(sa_ang_avg))
        #Rotate CSS sun vector from SA to ACA frame
        css_aca = np.array([sinang * data['aosunsa1'].vals -
                            cosang * data['aosunsa3'].vals,
                            data['aosunsa2'].vals,
                            cosang * data['aosunsa1'].vals +
                            sinang * data['aosunsa3'].vals])
        #Normalize sun vec (again) and compute pitch
        magnitude = sqrt((css_aca * css_aca).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        sun_vec_norm = css_aca / magnitude
        roll_css = degrees(arctan2(-sun_vec_norm[1, :], -sun_vec_norm[2, :]))
        return roll_css


#--------------------------------------------
class DP_ROLL_CSS_SA(DerivedParameterPcad):
    """Sun Roll Angle from CSS Data in SA Frame [Deg]

    Defined as the rotation about the SA-1 X-axis required to align the sun
    vector with the SA-1 X-Z plane.

    Calculated as ARCTAN( (-1*AOSUNSA2) / (-1*AOSUNSA3) ) using the
    four-quadrant version of ARCTAN.

    """
    rootparams = ['aosunsa2', 'aosunsa3']
    time_step = 8.2
    max_gap = 18.0
    dtype = np.float32

    def calc(self, data):
        roll_css_sa = degrees(arctan2(-data['aosunsa2'].vals,
                                      -data['aosunsa3'].vals))
        return roll_css_sa


#--------------------------------------------
class DP_ROLL_FSS(DerivedParameterPcad):
    """Off-Nominal Roll Angle from FSS Data in ACA Frame [Deg]

    Defined as the rotation about the ACA X-axis required to align the sun
    vector with the ACA X/Z plane.

    When in FSS FOV per AOSUNPRS:
    Calculated using the FSS alpha and beta angles to compute the sun vector
    in the FSS frame.  The sun vector is then rotated into the ACA frame
    using the rotation matrix (an OBC k-constant).  Roll is computed using the
    arctan function.

    When NOT in FSS FOV per AOSUNPRS:
    <data>.bads = 1
    """
    rootparams = ['aoalpang', 'aobetang', 'aosunprs']
    time_step = 1.025
    max_gap = 10.0
    dtype = np.float32

    def calc(self, data):
        in_fss_fov = (data['aosunprs'].vals == 'SUN ')
        data.bads = data.bads | ~in_fss_fov
        # rotation matrix from FSS to ACA frame
        A_AF = np.array([[9.999990450374580e-01,
                           0.0,
                          -1.382000062241829e-03],
                         [-5.327615067743422e-07,
                           9.999999256947376e-01,
                          -3.854999811959735e-04],
                         [1.381999959551952e-03,
                           3.855003493343671e-04,
                           9.999989707322665e-01]])
        # FSS's sun vector in FSS frame
        alpha = radians(data['aoalpang'].vals)
        beta = radians(data['aobetang'].vals)
        sun_fss = np.array([tan(beta), tan(alpha), -np.ones(len(alpha))])
        sun_aca = A_AF.dot(sun_fss)
        magnitude = sqrt((sun_aca * sun_aca).sum(axis=0))
        data.bads |= magnitude == 0.0
        magnitude[data.bads] = 1.0
        sun_vec_norm = sun_aca / magnitude
        roll_fss = degrees(arctan2(-sun_vec_norm[1, :], -sun_vec_norm[2, :]))
        return roll_fss


#--------------------------------------------
class DP_RW_MOM_TOT(DerivedParameterPcad):
    """Total Reaction Wheel Momentum [Ft-Lb-Sec]

    Defined as the RSS of AORWMOM1, AORWMOM2, and AORWMOM3.

    """
    rootparams = ['aorwmom1', 'aorwmom2', 'aorwmom3']
    time_step = 8.2
    dtype = np.float32

    def calc(self, data):
        rw_mom_tot = sqrt(data['aorwmom1'].vals ** 2 +
                             data['aorwmom2'].vals ** 2 +
                             data['aorwmom3'].vals ** 2)
        return rw_mom_tot


#--------------------------------------------
class DP_RW1_DELTA_TEMP(DerivedParameterPcad):
    """Difference between Reaction Wheel 1 Compartment and Bearing Temperature
    [Deg F]

    Defined as TCYZ_RW1 - ARWA1BT.

    """
    rootparams = ['tcyz_rw1', 'arwa1bt']
    time_step = 0.25625

    def calc(self, data):
        rw1_delta_temp = data['tcyz_rw1'].vals - data['arwa1bt'].vals
        return rw1_delta_temp


#--------------------------------------------
class DP_RW2_DELTA_TEMP(DerivedParameterPcad):
    """Difference between Reaction Wheel 2 Compartment and Bearing Temperature
    [Deg F]

    Defined as TPCP_RW2 - ARWA2BT.

    """
    rootparams = ['tpcp_rw2', 'arwa2bt']
    time_step = 0.25625

    def calc(self, data):
        rw2_delta_temp = data['tpcp_rw2'].vals - data['arwa2bt'].vals
        return rw2_delta_temp


#--------------------------------------------
class DP_RW3_DELTA_TEMP(DerivedParameterPcad):
    """Difference between Reaction Wheel 3 Compartment and Bearing Temperature
    [Deg F]

    Defined as TPCP_RW3 - ARWA3BT.

    """
    rootparams = ['tpcp_rw3', 'arwa3bt']
    time_step = 0.25625

    def calc(self, data):
        rw3_delta_temp = data['tpcp_rw3'].vals - data['arwa3bt'].vals
        return rw3_delta_temp


#--------------------------------------------
class DP_RW4_DELTA_TEMP(DerivedParameterPcad):
    """Difference between Reaction Wheel 4 Compartment and Bearing Temperature
    [Deg F]

    Defined as TPCM_RW4 - ARWA4BT.

    """
    rootparams = ['tpcm_rw4', 'arwa4bt']
    time_step = 0.25625

    def calc(self, data):
        rw4_delta_temp = data['tpcm_rw4'].vals - data['arwa4bt'].vals
        return rw4_delta_temp


#--------------------------------------------
class DP_RW5_DELTA_TEMP(DerivedParameterPcad):
    """Difference between Reaction Wheel 5 Compartment and Bearing Temperature
    [Deg F]

    Defined as TPCM_RW5 - ARWA5BT.

    """
    rootparams = ['tpcm_rw5', 'arwa5bt']
    time_step = 0.25625

    def calc(self, data):
        rw5_delta_temp = data['tpcm_rw5'].vals - data['arwa5bt'].vals
        return rw5_delta_temp


#--------------------------------------------
class DP_RW6_DELTA_TEMP(DerivedParameterPcad):
    """Difference between Reaction Wheel 6 Compartment and Bearing Temperature
    [Deg F]

    Defined as TCYZ_RW6 - ARWA6BT.

    """
    rootparams = ['tcyz_rw6', 'arwa6bt']
    time_step = 0.25625

    def calc(self, data):
        rw6_delta_temp = data['tcyz_rw6'].vals - data['arwa6bt'].vals
        return rw6_delta_temp


#--------------------------------------------
class DP_SA_ANG_AVG(DerivedParameterPcad):
    """Average Solar Array Angle [Deg]

    Defined as the mean of AOSARES1 and AOSARES2.

    """
    rootparams = ['aosares1', 'aosares2']
    time_step = 4.1
    max_gap = 10.0

    def calc(self, data):
        sa_ang_avg = (1.0 * data['aosares1'].vals +
                      1.0 * data['aosares2'].vals) / 2
        return sa_ang_avg


#--------------------------------------------
class DP_SUN_XZ_ANGLE(DerivedParameterPcad):
    """Angle between Sun and ACA X/Z plane [Deg]

    Incidence angle of the Sun vector on the ACA X/Z plane.

    Calculated using the four-quadrant arctan of the sun vector y and z
    components in the ACA frame where the sun vector is from definitive
    ephemeris [SOLAREPHEM0 and ORBITEPHEM0] and the estimated attitude from
    the OBC's estimated quaternion [AOATTQT<n>].

    http://occweb.cfa.harvard.edu/twiki/pub/Aspect/WebHome/ROLLDEV3.pdf
    """
    rootparams = ['orbitephem0_x', 'orbitephem0_y', 'orbitephem0_z',
                  'solarephem0_x', 'solarephem0_y', 'solarephem0_z',
                  'aoattqt1', 'aoattqt2', 'aoattqt3', 'aoattqt4']
    time_step = 1.025
    max_gap = 4.0
    max_gaps = {msid: 602.0 for msid in rootparams if 'ephem' in msid}
    dtype = np.float32

    def calc(self, data):
        sun_vec_b = sun_vector_body(data)
        sun_xz_angle = degrees(arctan2(sun_vec_b[1],
                                       sqrt(sun_vec_b[0] ** 2 +
                                            sun_vec_b[2] ** 2)))
        return sun_xz_angle


#--------------------------------------------
class DP_SYS_MOM_TOT(DerivedParameterPcad):
    """Total System Momentum [Ft-Lb-Sec]

    Defined as the sum of the reaction wheel, environmental, and spacecraft
    momentum.

    Calculated as the RSS of AOSYMOM1, AOSYMOM2, and AOSYMOM3.

    """
    rootparams = ['aosymom1', 'aosymom2', 'aosymom3']
    time_step = 8.2
    max_gap = 18.0

    def calc(self, data):
        sys_mom_tot = sqrt(data['aosymom1'].vals ** 2 +
                           data['aosymom2'].vals ** 2 +
                           data['aosymom3'].vals ** 2)
        return sys_mom_tot


#--------------------------------------------
def qmult(q1, q2):
    """Multiply two quaternions or arrays of quaternions

    The input quaternions must have shape of (4,) or (4, N, ..).

    :param q1: first quaternion
    :param q2: second quaternion
    :returns: q1*q2 as an array with same shape as q1 and q2
    """
    if q1.shape != q2.shape:
        raise ValueError('Shapes must agree')

    mult = np.zeros_like(q1)
    mult[0] = q1[3] * q2[0] - q1[2] * q2[1] + q1[1] * q2[2] + q1[0] * q2[3]
    mult[1] = q1[2] * q2[0] + q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3]
    mult[2] = -q1[1] * q2[0] + q1[0] * q2[1] + q1[3] * q2[2] + q1[2] * q2[3]
    mult[3] = -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3]

    return mult


#--------------------------------------------
def qrotate(q, r):
    """Rotate a vector by a quaternion

    The input quaternion must have a shape of (4,) or (4, N, ..).

    The input vector must have a shape of (3,) or (3, N, ..).

    :param q:  quaternion defining the rotation
    :param r:  vector to be rotated

    :returns r rotated by q as an array with the same shape as r
    """
    if q.shape[0] != 4:
        raise ValueError('Input quaternion must have shape (4,) or (4, N, ..)')
    if r.shape[0] != 3:
        raise ValueError('Input vector must have shape (3,) or (3, N, ..).')
    rot = np.zeros_like(r)
    rot[0] = (r[0] * (q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2) +
              r[1] * (q[0] * q[1] + q[2] * q[3]) * 2 +
              r[2] * (q[0] * q[2] - q[1] * q[3]) * 2)
    rot[1] = (r[0] * (q[0] * q[1] - q[2] * q[3]) * 2 -
              r[1] * (q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2) +
              r[2] * (q[1] * q[2] + q[0] * q[3]) * 2)
    rot[2] = (r[0] * (q[0] * q[2] + q[1] * q[3]) * 2 +
              r[1] * (q[1] * q[2] - q[0] * q[3]) * 2 -
              r[2] * (q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2))

    return rot


def arccos_clip(x):
    return np.arccos(x.clip(-1, 1))


def sun_vector_body(data, predictive=True):
    """Calculate the normalized sun vector in body coordinates.

    :param data: MSIDset with orbitephem, solarephem and aoattqt<N> MSIDs
    :param predictive: use predictive ephemeris
    :returns: 3 x N array of vectors
    """
    orbit = 'orbitephem{}_'.format('0' if predictive else '1')
    solar = 'solarephem{}_'.format('0' if predictive else '1')

    chandra_eci = np.array([data[orbit + 'x'].vals,
                            data[orbit + 'y'].vals,
                            data[orbit + 'z'].vals])
    sun_eci = np.array([data[solar + 'x'].vals,
                        data[solar + 'y'].vals,
                        data[solar + 'z'].vals])
    sun_vec = -chandra_eci + sun_eci
    est_quat = np.array([data['aoattqt1'].vals,
                         data['aoattqt2'].vals,
                         data['aoattqt3'].vals,
                         data['aoattqt4'].vals])

    sun_vec_b = qrotate(est_quat, sun_vec)  # Rotate into body frame
    magnitude = sqrt((sun_vec_b ** 2).sum(axis=0))
    data.bads |= magnitude == 0.0
    magnitude[data.bads] = 1.0
    sun_vec_b = sun_vec_b / magnitude  # Normalize

    return sun_vec_b
