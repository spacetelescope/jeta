========================================
Pseudo-MSIDs in the engineering archive
========================================

A small selection of pseudo-MSIDs that do not come in the engineering telemetry
stream are also available in the archive.  These are:

* ACIS DEA housekeeping: status from the DEA (including detector focal plane temperature)
* Ephemeris: predictive and definitive orbital (Chandra), solar, and lunar ephemeris values
* SIM telemetry: SIM position and moving status
* Derived parameters: values computed from other MSIDs in the archive

ACIS DEA housekeeping
--------------------------------------------------

The ACIS DEA telemeters a variety of useful information that is sent
in an event-based format via queries to the processor.  The engineering
archive reformats those telemetry queries (one per psuedo-MSID) into
records that match the engineering archive format where all queries with the
same time stamp are place in a single record.

The time sample of these data vary but are typically around once per 16 seconds.
Because of what appears to be an issue with CXCDS decom there are frequently
bad values at the beginning of an archive file.  For this reason it is
especially important to perform a fetch with the ``filter_bad=True`` setting.

==================== ====== =======================================
MSID                 Unit   Description
==================== ====== =======================================
tmp_bep_pcb          K      DPA Thermistor 1 - BEP PC Board
tmp_bep_osc          K      DPA Thermistor 2 - BEP Oscillator
tmp_fep0_mong        K      DPA Thermistor 3 - FEP 0 Mongoose
tmp_fep0_pcb         K      DPA Thermistor 4 - FEP 0 PC Board
tmp_fep0_actel       K      DPA Thermistor 5 - FEP 0 ACTEL
tmp_fep0_ram         K      DPA Thermistor 6 - FEP 0 RAM
tmp_fep0_fb          K      DPA Thermistor 7 - FEP 0 Frame Buf
tmp_fep1_mong        K      DPA Thermistor 8 - FEP 1 Mongoose
tmp_fep1_pcb         K      DPA Thermistor 9 - FEP 1 PC Board
tmp_fep1_actel       K      DPA Thermistor 10 - FEP 1 ACTEL
tmp_fep1_ram         K      DPA Thermistor 11 - FEP 1 RAM
tmp_fep1_fb          K      DPA Thermistor 12 - FEP 1 Frame Buf
fptemp_12            K      Focal Plane Temp. Board 12
fptemp_11            K      Focal Plane Temp. Board 11
dpagndref1           V      DPA Ground Reference 1
dpa5vhka             V      DPA 5V Housekeeping A
dpagndref2           V      DPA Ground Reference 2
dpa5vhkb             V      DPA 5V Housekeeping B
dea28volta           V      Primary Raw DEA 28V DC
dea24volta           V      Primary Raw DEA 24V DC
deam15volta          V      Primary Raw DEA -15.5V
deap15volta          V      Primary Raw DEA +15.5V
deam6volta           V      Primary Raw DEA -6V DC
deap6volta           V      Primary Raw DEA +6V DC
rad_pcb_a                   Relative Dose Rad. Monitor Side A
gnd_1                V      Interface Ground Reference
dea28voltb           V      Backup Raw DEA 28V DC
dea24voltb           V      Backup DEA 24V DC
deam15voltb          V      Backup DEA -15.5V DC
deap15voltb          V      Backup DEA +15.5V DC
deam6voltb           V      Backup DEA -6V DC
deap6voltb           V      Backup DEA +6V DC
rad_pcb_b                   Relative Dose Rad. Monitor Side B
gnd_2                V      Ground
==================== ====== =======================================

Ephemeris
---------

CXC processing generates definitive and predictive ephemeris files for the
Chandra, the Moon, and the Sun.  Values are given with respect to Earth (ECI).
Predictive values are available into the near future while definitive values
will be a few weeks behind the present.  (Note that daily and 5-minute
statistics are only available up to the present time).  These values are
contained within the following content types:

============ ======== ===========
Content      Object   Type
============ ======== ===========
orbitephem0  Chandra  Predictive
lunarephem0  Moon     Predictive
solarephem0  Sun      Predictive
orbitephem1  Chandra  Definitive
lunarephem1  Moon     Definitive
solarephem1  Sun      Definitive
============ ======== ===========

The psuedo-MSIDs for each of the ephemeris elements is given in the
following table, where <CONTENT> is replaced by the appropriate Content value
from the previous table.

==================== ====== =================
MSID                 Unit   Description
==================== ====== =================
<CONTENT>_x          m      X position (ECI)
<CONTENT>_y          m      Y position (ECI)
<CONTENT>_z          m      Z position (ECI)
<CONTENT>_vx         m/s    X velocity (ECI)
<CONTENT>_vy         m/s    Y velocity (ECI)
<CONTENT>_vz         m/s    Z velocity (ECI)
==================== ====== =================

In addition there is a set of pseudo-MSIDs that provide a number of
higher-level definitive angle and distance values that are useful.

==================== ====== =========================================
MSID                 Unit   Description
==================== ====== =========================================
Point_X                     Unit Pointing (X)
Point_Y                     Unit Pointing (Y)
Point_Z                     Unit Pointing (Z)
Point_SunCentAng     deg    Pointing-Solar angle (from center)
Point_SunLimbAng     deg    Pointing-Solar angle (from limb)
Point_MoonCentAng    deg    Pointing-Lunar angle (from center)
Point_MoonLimbAng    deg    Pointing-Lunar angle (from limb)
Point_EarthCentAng   deg    Pointing-Earth angle (from center)
Point_EarthLimbAng   deg    Pointing-Earth angle (from limb)
Dist_SatEarth        m      Sat-Earth distance (from Earth center)
Sun_EarthCentAng     deg    Sun-Earth angle (from center)
Sun_EarthLimbAng     deg    Sun-Earth angle (from limb)
Point_RamVectorAng   deg    Pointing-Ram angle
==================== ====== =========================================

HRC Secondary Science and Housekeeping
--------------------------------------

The HRC telemeters a variety of useful information that is sent down in secondary science
and housekeeping formats (SS and HK hereafter).  The engineering archive reformats those
telemetry values to be consistent with the MSFC-1949 specification of HRC SS and HK
telemetry.  It also facilitates handling invalid data in SS and HK telemetry.

Invalid data
^^^^^^^^^^^^

Invalid HRC SS and HK telemetry can arise in three ways:

1. When telemetry format changes there is commanding to change the timing signals used to
   fill the housekeeping and secondary science rates that can result in invalid data being
   put in these MSIDs for up to a major frame.
2. When detectors change or a detector is set to its default configuration the FIFO used
   to hold the housekeeping an secondary science data gets reset which may result in a single
   bad sample of data.
3. The secondary-science byte-shift anomaly causes the occasional portion of the
   housekeeping and sometimes the rate data to be corrupted.

In the Ska archive the presence of these conditions is tracked in a new pseudo-MSID called
``HRC_SS_HK_BAD``.

The first two of these are detected by looking at "spare" bits in the MSID ``SCIDPREN``
(i.e. good data satisfies ``SCIDPREN=0000xxxx000xxxxx``.  The least-significant 7 bits of
``HRC_SS_HK_BAD`` contain a copy of the 7 bits in ``SCIDPREN`` which must be 0 for good
data.  The following code illustrates detecting conditions (1) or (2)::

  >>> from Ska.engarchive import fetch
  >>> from Chandra.Time import DateTime
  >>> dat = fetch.Msid('HRC_SS_HK_BAD', '1999:300', '1999:310')
  >>> bad = (dat.vals & 0x7f) > 0
  >>> DateTime(dat.times[bad]).date
  array(['1999:301:16:10:13.375', '1999:301:16:10:15.425',
         '1999:301:18:16:42.476', '1999:301:18:16:44.526',
         '1999:301:19:20:03.176', '1999:301:19:20:05.226',
         '1999:301:21:28:33.226', '1999:301:21:28:35.276',
         '1999:303:07:13:16.230', '1999:303:07:13:18.280'],
        dtype='|S21')

The third condition is detected by its impact on the MSID ``2SMTRATM``. If it is less than
-20degC or greater than 50degC then the analog housekeeping from the row is
marked with bad quality.  In this case the ``HRC_SS_HK_BAD`` MSID has bit 10 set,
which can be detected by a logical-and with ``0x0400`` (1024).

Querying data
^^^^^^^^^^^^^^

For HK telemetry it is sufficient to query the archive using the standard ``fetch.Msid``
method which automatically removes bad quality data.  This applies for 5-minute and daily
stat data as well.  For instance::

  >>> dat = fetch.Msid('2S2ONST', '2002:200', '2002:250')
  >>> dat.plot()

.. plot::

   from Ska.engarchive import fetch
   import matplotlib.pyplot as plt
   plt.figure(figsize=(6, 4), dpi=75)
   dat = fetch.Msid('2S2ONST', '2002:200', '2002:250', stat='5min')
   dat.plot()
   plt.grid()
   plt.tight_layout()

For SS the situation is a little different because those do not have
the bad quality flags set at the time of data ingest (because the indicators are all
in HK).  In this case use the ``fetch.HrcSsMsid`` method to get a filtered version
of the SS MSIDs (``2TLEV1RT 2VLEV1RT 2SHEV1RT 2TLEV2RT 2VLEV2RT 2SHEV2RT``).  For
instance to get 5-minute telemetry for ``2SHEV1RT`` use::

  >>> dat = fetch.HrcSsMsid('2SHEV1RT', '2002:200', '2002:250', stat='5min')
  >>> dat.plot()

.. plot::

   from Ska.engarchive import fetch
   import matplotlib.pyplot as plt
   plt.figure(figsize=(6, 4), dpi=75)
   dat = fetch.HrcSsMsid('2SHEV1RT', '2002:200', '2002:250', stat='5min')
   dat.plot()
   plt.grid()
   plt.tight_layout()

HK MSIDs
^^^^^^^^

The list of available HK MSIDs is:

=========    ====================================================
MSID          Description
=========    ====================================================
224PCAST     +24V LVPS ON/OFF
215PCAST     +15V LVPS ON/OFF
215NCAST     -15V LVPS ON/OFF
2SPTPAST     SPECTROSCOPY DET TOP PLATE HV STEP
2SPBPAST     SPECTROSCOPY DET BOTTOM PLATE HV STEP
2IMTPAST     IMAGING DET TOP PLATE HV STEP
2IMBPAST     IMAGING DET BOTTOM PLATE HV STEP
2NYMTAST     -Y SHUTTER MOTOR SELECTED
2PYMTAST     +Y SHUTTER MOTOR SELECTED
2CLMTAST     CALSRC MOTOR SELECTED
2DRMTAST     DOOR MOTOR SELECTED
2ALMTAST     ALL MOTORS DESELECTED
2MSMDARS     MOTION CONTROL MODE RESET -- 2MSMDARS
2MDIRAST     MOTOR DIRECTION
2MSNBAMD     MOTOR STATUS REGISTER MV NSTEPS TOWARD B
2MSNAAMD     MOTOR STATUS REGISTER MV NSTEPS TOWARD A
2MSLBAMD     MOTOR STATUS REGISTER MOVE TO LIMIT SWITCH B
2MSLAAMD     MOTOR STATUS REGISTER MOVE TO LIMIT SWITCH A
2MSPRAMD     MOTOR STATUS REGISTER MOVE TO POSITION R
2MSDRAMD     MOTOR DRIVE ENABLE
2MCMDARS     MOTION CONTROL MODE RESET -- 2MCMDARS
2MCNBAMD     MOTOR CMD REGISTER MV NSTEPS TOWARD B
2MCNAAMD     MOTOR CMD REGISTER MV NSTEPS TOWARD A
2MCLBAMD     MOTOR CMD REGISTER MOVE TO LIMIT SWITCH B
2MCLAAMD     MOTOR CMD REGISTER MOVE TO LIMIT SWITCH A
2MCPRAMD     MOTOR COMMAND REGISTER MOVE TO POSITION REGISTER
2MDRVAST     MOTOR CMD REGISTER MOTOR DRIVE ENABLE
2SCTHAST     STEP CTR LAST VALUE
2SMOIAST     SELECTED MOTOR OVERCURRENT FLAG
2SMOTAST     SELECTED MOTOR OVERTEMPERATURE FLAG
2DROTAST     DRV OVERTEMP ENABLE
2DROIAST     DRV OVERCURRENT ENABLE
2SFLGAST     STOP FLAG ENABLE
2OSLSAST     OPEN SECONDARY LIMIT SWITCH ENABLE
2OPLSAST     OPEN PRIMARY LIMIT SWITCH ENABLE
2CSLSAST     CLOS SECONDARY LIMIT SWITCH ENABLE
2CPLSAST     CLOS PRIMARY LIMIT SWITCH ENABLE
2OSLSADT     OPEN SECONDARY LS DETECTED
2OSLSAAC     OPEN SECONDARY LS ACTIVE
2OPLSAAC     OPEN PRIMARY LS ACTIVE
2CSLSADT     CLOS SECONDARY LS DETECTED
2CSLSAAC     CLOS SECONDARY LS ACTIVE
2CPLSAAC     CLOS PRIMARY LS ACTIVE
2FCPUAST     FORCED COARSE POSITION U AXIS
2FCPVAST     FORCED COARSE POSITION V AXIS
2CBHUAST     CENTER BLANK HIGH CP U AXIS
2CBLUAST     CENTER BLANK LOW CP U AXIS
2CBHVAST     CENTER BLANK HIGH CP V AXIS
2CBLVAST     CENTER BLANK LOW CP V AXIS
2WDTHAST     WIDTH THRESHOLD SETTING
2CLMDAST     CALIBRATION MODE ON
2FIFOAVR     DATA FIFO ENABLE
2OBNLASL     OBSERVING/NEXT-IN-LINE MODE SELECT
2SPMDASL     SPECT DETECTOR SPECT/IMG MODE SELECT
2EBLKAVR     EDGE BLANK VALIDITY ENABLE
2CBLKAVR     CENTER BLANK VALIDITY ENABLE
2ULDIAVR     UPPER LEVEL DISCR VALIDITY ENABLE
2WDTHAVR     WIDTH DISCR VALIDITY ENABLE
2SHLDAVR     SHIELD DISCR VALIDITY ENABLE
2SPONST      SPECTROSCOPY DETECTOR HVPS ON/OFF
2SPCLST      SPECTROSCOPY DET HVPS CURRENT LIMIT ENAB
2S1ONST      SHIELD A HVPS ON/OFF
2IMONST      IMAGING DETECTOR HVPS ON/OFF
2IMCLST      IMAGING DET HVPS CURRENT LIMIT ENABLE
2S2ONST      SHIELD B HVPS ON/OFF
2S1HVST      SHIELD A HVPS SETTING
2S2HVST      SHIELD B HVPS SETTING
2C05PALV     +5V BUS MONITOR
2C15PALV     +15V BUS MONITOR
2C15NALV     -15V BUS MONITOR
2C24PALV     +24V BUS MONITOR
2IMHVLV      IMAGING LOWER MCP HV MONITOR
2IMHBLV      IMAGING LOWER & UPPER MCP HV MONITOR
2SPHVLV      SPECTROSCOPY LOWER MCP HV MONITOR
2SPHBLV      SPECTROSCOPY UPPER MCP HV MONITOR
2S1HVLV      SHIELD A HV MONITOR
2S2HVLV      SHIELD B HV MONITOR
2PRBSCR      PRIMARY BUS CURRENT
2PRBSVL      PRIMARY BUS VOLTAGE
2ULDIALV     UPPER LEVEL DISCRIMINATOR SETTING
2LLDIALV     TRIGGER LEVEL DISCRIMINATOR MONITOR
2FEPRATM     FE PREAMP CARD TEMPERATURE
2CALPALV     CAL PULSER AMPLITUDE MONITOR
2GRDVALV     GRID BIAS SETTING MONITOR
2RSRFALV     RANGE SWITCH ANALOG SETTING
2SPINATM     SPECTROSCOPY DETECTOR TEMPERATURE (INSIDE)
2IMINATM     IMAGING DETECTOR TEMPERATURE (INSIDE)
2LVPLATM     LVPS PLATE TEMP
2SPHVATM     SPECTROSCOPY DET HVPS TEMPERATURE
2IMHVATM     IMAGING DET HVPS TEMPERATURE
2SMTRATM     SELECTED MOTOR TEMPERATURE
2FE00ATM     FRONT END TEMPERATURE RT2
=========    ====================================================

SS MSIDs
^^^^^^^^
The available SS MSIDs are:

=========    ====================================================
MSID          Description
=========    ====================================================
2TLEV1RT     TOTAL EVENT RATE 1
2VLEV1RT     VALID EVENT RATE 1
2SHEV1RT     SHIELD EVENT RATE 1
2TLEV2RT     TOTAL EVENT RATE 2
2VLEV2RT     VALID EVENT RATE 2
2SHEV2RT     SHIELD EVENT RATE 2
=========    ====================================================


Science Instrument Module
-------------------------

Information about the SIM is available via the three following pseudo-MSIDs
categories.

SEA standard telemetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The units shown below are for the CXC and ENG unit systems, respectively.

============ ========= ===============================================
MSID         Unit      Description
============ ========= ===============================================
3FAFLAAT     K [degC]  SEA FA flexure a temp a
3FAFLBAT     K [degC]  SEA FA flexure b temp a
3FAFLCAT     K [degC]  SEA FA flexure c temp a
3FAMOVE                SEA FA in motion flag
3FAMTRAT     K [degC]  SEA-A focus drive motor temp
3FAPOS       mm [step] SEA FA position
3FAPSAT      K [degC]  SEA-A power supply temp
3FASEAAT     K [degC]  SEA-A box temp
3LDRTMEK               SEA mechanism for last detected reference tab
3LDRTNO                SEA tab number of reference tab last detected
3LDRTPOS     mm [step] SEA last detected ref tab position
3MRMMXMV        [step] SEA max pwm level most recent move
3SEAID                 SEA identification
3SEAINCM               SEA invalid command group flag
3SEARAMF               SEA ram failure detection flag
3SEAROMF               SEA prom checksum fail flag
3SEARSET               SEA reset flag
3SEATMUP               SEA tlm update flag (toggle w/ea update)
3SFLXAST     K [degC]  SEA flexure a temperature setpoint
3SFLXBST     K [degC]  SEA flexure b temperature setpoint
3SFLXCST     K [degC]  SEA flexure c temperature setpoint
3SHTREN                SEA heater power relay status
3SMOTOC      cnts      SEA motor drive overcurrent counter
3SMOTPEN               SEA motor driver power relay status
3SMOTSEL               SEA motor selection relay status
3SMOTSTL     cnts      SEA motor stall counter
3SPENDC      cnts      SEA pending cmd count
3STAB2EN               SEA tab2 auto position update enab/disa status
3TRMTRAT     K [degC]  SEA a translation drive motor temp
3TSCMOVE               SEA TSC in motion flag
3TSCPOS      mm [step] SEA TSC position
TLMSTATUS              SEA telemetry status (updated or not updated)
============ ========= ===============================================

The state codes for these MSIDs (where applicable) are defined by the CXC `SIM level-0
decom specification <http://icxc.harvard.edu/icd/Sim/Level0/1.2/l0icd.ps>`_ and differ
from the values found in the TDB.  The engineering archive state codes are:

======== ======= ========
MSID     Raw=0   Raw=1
======== ======= ========
3TSCMOVE F       T
3FAMOVE  F       T
3SEAID   SEA-A   SEA-B
3SEARSET F       T
3SEAROMF F       T
3SEAINCM F       T
3STAB2EN DISABLE ENABLE
3SMOTPEN ENABLE  DISABLE
3SMOTSEL TSC     FA
3SHTREN  DISABLE ENABLE
3SEARAMF F       T
======== ======= ========

SEA diagnostic telemetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============ ========= ===============================================
MSID         Unit      Description
============ ========= ===============================================
3SDSWELF               SEA CSC Exectuting from RAM
3SDPSTKP               SEA Data Stack Ptr
3SDTSEDG               TSC Tab Edge Detection Flags
3SDFAEDG               FA Tab Edge Detection Flags
3SDMAJFP               Major Frame Period Time Measured by SEA
3SDRMOVD               Most Recent Motor Move Destination
3SDTSTSV     V         TSC Tab Position Sensor A/D converter
3SDFATSV     V         FA Tab Position Sensor A/D Converter
3SDAGV       V         Analog Ground A/D Converter Reading
3SDP15V      V         +15V Power Supply A/D Converter Reading
3SDP5V       V         +5V Power Supply A/D Converter Reading
3SDM15V      V         -15V Power Supply A/D Converter Reading
3SDFLXAT     K [degC]  Flexure A Thermistor A/D Converter
3SDFLXBT     K [degC]  Flexure B Thermistor A/D Converter
3SDFLXCT     K [degC]  Flexure C Thermistor A/D Converter
3SDTSMT      K [degC]  TSC Motor Thermistor A/D Converter
3SDFAMT      K [degC]  FA Motor Thermistor A/D Converter
3SDPST       K [degC]  SEA Power Supply Thermistor A/D Converter
3SDBOXT                SEA Box Thermistor A/D Converter
3SDRMFAD               RAM Most Recent detected Fail Address
3SDTSTBW               TSC Most Recent detected Tab Width
3SDFATBW               FA Most Recent detected Tab Width
3SDSYRS                Process Reset Due Synchronization Loss
3SDWMRS                Processor Warm Reset
3SDTSP                 TSC Most Recent PWM Histogram
3SDFAP                 FA Most Recent PWM Histogram
3SDINCOD               SEA Invalid CommandCode
============ ========= ===============================================

The state codes for these MSIDs (where applicable) are defined by the CXC `SIM level-0
decom specification <http://icxc.harvard.edu/icd/Sim/Level0/1.2/l0icd.ps>`_ and differ
from the values found in the TDB.  The engineering archive state codes are:

======== ======= ========
MSID     Raw=0   Raw=1
======== ======= ========
3SDSWELF F       T
3SDSYRS  F       T
3SDWMRS  F       T
======== ======= ========

SIMCOOR (CXC high-level values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. Note::  These pseudo-MSIDs are deprecated in favor of the standard
   versions such as 3TSCPOS, 3FAPOS, 3TSCMOV, 3TRMTRAT, etc. which
   are available in the SEA telemetry described above.

==================== ====== =========================================
MSID                 Unit   Description
==================== ====== =========================================
SEAIDENT                     SEA identification
SIM_X                mm      X position (FA)
SIM_Y                mm      Y position (not meaningful)
SIM_Z                mm      Z position (TSC)
SIM_X_MOVED                  FA moved
SIM_Z_MOVED                  TSC moved
==================== ====== =========================================

EPHIN
------

Information about the EPHIN instrument is available via the following pseudo-MSIDs:

==================== ====== =========================================
MSID                 Unit   Description
==================== ====== =========================================
TLMBLKCNT                    EIO TLMBLK count
EIOBITCNT                    EIO bit counter
HKOPMODE                     HK operational Mode
HKRESET                      HK reset Flag
HKDOWNLOAD                   HK down load flag
HKUPLOAD                     HK upload Flag
HKFRAMECNTR                  HK internal frame Counter
HKRINGSEGW                   HK ring segment auto switching
HKFAILMODEA                  HK failure mode detector A
HKFAILMODEB                  HK failure mode detector B
HKHVDETG                     HK high voltage detector G
HKHVDETAF                    HK high voltage detectors A-F
HKANALOGPWR                  HK analog power switchs
HKFAILMODEGC                 HK failure mode detectors G-C
HKP5V                 V      HK +5V rail voltage
HKP27V                V      HK +27V rail voltage
HKP6V                 V      HK +6V rail voltage
HKN6V                 V      HK -6V rail voltage
HKP5I                 mA     HK +5V rail current
HKP27I                mA     HK +27V rail current
HKP6I                 mA     HK +6V rail current
HKN6I                 mA     HK -6V rail current
HKEBOXTEMP            K      HK EBox temperature (5EHSE300)
HKABIASLEAKI          uA     HK A bias leakage current
HKBBIASLEAKI          uA     HK B bias leakage current
HKCBIASLEAKI          uA     HK C bias leakage current
HKDBIASLEAKI          uA     HK D bias leakage current
HKEBIASLEAKI          uA     HK E bias leakage current
HKFBIASLEAKI          uA     HK F bias leakage current
HKGHV                 V      HK G high voltage
SCOPMODE                     Sci operational mode
SCSTATUS                     Sci status flags
SCFRAMECNTR                  Sci internal Frame Counter
SCCONTROL                    Sci control flags
SCRINGSEGSW                  Sci ring segment auto switching
SCFAILMODEA                  Sci failure mode detectors A
SCFAILMODEB                  Sci failure mode detectors B
SCHVDETG                     Sci high voltage detector G
SCHVDETAF                    Sci high voltage detectors A-F
SCANALOGPWR                  Sci analog power switches
SCFAILMODEGC                 Sci failure mode detectors G-C
SCPHAPRIPTR                  Sci PHA Priority pointer
SCG0                         Sci single detector counter G0
SCA00                        Sci single detector counter A00
SCA01                        Sci single detector counter A01
SCA02                        Sci single detector counter A02
SCA03                        Sci single detector counter A03
SCA04                        Sci single detector counter A04
SCA05                        Sci single detector counter A05
SCB00                        Sci single detector counter B00
SCB01                        Sci single detector counter B01
SCB02                        Sci single detector counter B02
SCB03                        Sci single detector counter B03
SCB04                        Sci single detector counter B04
SCB05                        Sci single detector counter B05
SCC0                         Sci single detector counter C0
SCD0                         Sci single detector counter D0
SCE0                         Sci single detector counter E0
SCF0                         Sci single detector counter F0
SCP4GM                       Sci single detector counter P4GM
SCP4GR                       Sci single detector counter P4GR
SCP4S                        Sci single detector counter P4S
SCP8GM                       Sci single detector counter P8GM
SCP8GR                       Sci single detector counter P8GR
SCP8S                        Sci single detector counter P8S
SCH4GM                       Sci single detector counter H4GM
SCH4GR                       Sci single detector counter H4GR
SCH4S1                       Sci single detector counter H4S1
SCH4S23                      Sci single detector counter H4S23
SCH8GM                       Sci single detector counter H8GM
SCH8GR                       Sci single detector counter H8GR
SCH8S1                       Sci single detector counter H8S1
SCH8S23                      Sci single detector counter H8S23
SCE150                       Sci single detector counter E150
SCE300                       Sci single detector counter E300
SCE1300                      Sci single detector counter E1300
SCE3000                      Sci single detector counter E3000
SCINT                        Sci single detector counter INT
SCP25GM                      Sci single detector counter P25GM
SCP25GR                      Sci single detector counter P25GR
SCP25S                       Sci single detector counter P25S
SCP41GM                      Sci single detector counter P41GM
SCP41GR                      Sci single detector counter P41GR
SCP41S                       Sci single detector counter P41S
SCH25GM                      Sci single detector counter H25GM
SCH25GR                      Sci single detector counter H25GR
SCH25S1                      Sci single detector counter H25S1
SCH25S23                     Sci single detector counter H25S23
SCH41GM                      Sci single detector counter H41GM
SCH41GR                      Sci single detector counter H41GR
SCH41S1                      Sci single detector counter H41S1
SCH41S23                     Sci single detector counter H41S23
SCCT0                        Sci single detector counter CT0
SCCT1                        Sci single detector counter CT1
SCCT2                        Sci single detector counter CT2
SCCT3                        Sci single detector counter CT3
SCCT4                        Sci single detector counter CT4
SCCT5                        Sci single detector counter CT5
==================== ====== =========================================


Derived Parameters
------------------

The engineering archive has pseudo-MSIDs that are derived via computation from
telemetry MSIDs.  All derived parameter names begin with the characters ``DP_``
(not case sensitive as usual).  Otherwise there is no difference from standard
MSIDs.

Definition
^^^^^^^^^^^

Derived parameters are defined by inheriting from the ``DerivedParameter`` base
class.  Each class definition requires three class attributes:
``content_root``, ``rootparams``, and ``time_step``.  The ``time_step`` should
be an integral multiple of 0.25625.  In the example below a large number of
definition classes have the same content root so another class
``DerivedParameterThermal`` has been created to avoid repeating the
``content_root`` definition every time.

Each definition class also requires a ``calc(self, data)`` method.  The
``data`` argument will be an MSIDset (dict of fetch MSID objects) with
values for each of the ``rootparams`` MSIDs.  The data values in the
MSIDset will be filtered for bad values and aligned to a common time
sequence with step size ``time_step``.
::

  class DerivedParameterThermal(base.DerivedParameter):
      content_root = 'thermal'

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

  class DP_P01(DerivedParameterThermal):
      """Zone 1 heater power"""
      rootparams = ['ELBV', '4OHTRZ01']
      time_step = 0.25625

      def calc(self, data):
          VSQUARED = data['ELBV'].vals * data['ELBV'].vals
          P01 = data['4OHTRZ01'].vals * VSQUARED / 110.2
          return P01

  class DP_DPA_POWER(base.DerivedParameter):
      """ACIS total DPA-A and DPA-B power"""
      rootparams = ['1dp28avo', '1dpicacu', '1dp28bvo', '1dpicbcu']
      time_step = 32.8
      content_root = 'acispow'

      def calc(self, data):
          power = (data['1dp28avo'].vals * data['1dpicacu'].vals +
                   data['1dp28bvo'].vals * data['1dpicbcu'].vals)
          return power


ACIS Power
^^^^^^^^^^^
.. automodule:: Ska.engarchive.derived.acispow
   :members:

EPS
^^^^^^^^^^^^^^^^^
.. automodule:: Ska.engarchive.derived.eps
   :members:
   :undoc-members:

Orbital elements
^^^^^^^^^^^^^^^^^
.. automodule:: Ska.engarchive.derived.orbit
   :members:
   :undoc-members:

PCAD
^^^^^
.. automodule:: Ska.engarchive.derived.pcad
   :members:

Thermal
^^^^^^^^^
.. automodule:: Ska.engarchive.derived.thermal
   :members:
   :undoc-members:

