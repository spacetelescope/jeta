#!/bin/bash

# This script allows controlled truncation and reprocessing of the live Ska engineering
# archive data.  The loop structure is used so that only one content type is truncated and
# reprocessed at once, thus minimizing impact to users and reducing the footprint in case
# something goes wrong.
#
# To use this, most likely you will first want to do a test where CONTENTS is set to just
# one type, for instance misc8eng is a good one because it is very small.
#
# Usage (from the dev_utils directory within a persistent NX window):
#
# $ ./repro_interval.sh
#
# This will slowly go through each content type, truncate and re-ingest.  Open another
# window and `tail -f trunc.log`.

# The following have been removed from the CONTENTS list below because quite often
# they don't need repro in the same way as the others.  But put them back in if
# it makes sense.
#
# lunarephem0
# solarephem0
# orbitephem1
# lunarephem1
# solarephem1
# angleephem

CONTENTS="
acisdeahk
acis2eng
acis3eng
acis4eng
ccdm1eng
ccdm2eng
ccdm3eng
ccdm4eng
ccdm5eng
ccdm7eng
ccdm8eng
ccdm10eng
ccdm11eng
ccdm12eng
ccdm13eng
ccdm14eng
ccdm15eng
ephin1eng
ephin2eng
eps1eng
eps2eng
eps3eng
eps4eng
eps5eng
eps6eng
eps7eng
eps9eng
eps10eng
hrc0hk
hrc0ss
hrc2eng
hrc4eng
hrc5eng
misc1eng
misc2eng
misc3eng
misc4eng
misc5eng
misc6eng
misc7eng
misc8eng
obc3eng
obc4eng
obc5eng
pcad3eng
pcad4eng
pcad5eng
pcad6eng
pcad7eng
pcad8eng
pcad10eng
pcad11eng
pcad12eng
pcad13eng
pcad14eng
pcad15eng
prop1eng
prop2eng
sim1eng
sim2eng
sim3eng
sim21eng
sms1eng
sms2eng
tel1eng
tel2eng
tel3eng
thm1eng
thm2eng
thm3eng
simdiag
simcoor
sim_mrg
ephhk
cpe1eng
dp_thermal1
dp_thermal128
dp_acispow128
dp_pcad1
dp_pcad4
dp_pcad16
dp_pcad32
dp_orbit1280
dp_eps8
dp_eps16"

touch trunc.log

TRUNCATE=2017:018
DATAROOT=/proj/sot/ska/data/eng_archive
# DRY="--dry-run"
DRY=""

for CONTENT in $CONTENTS
do
  echo ""

  CMD="/proj/sot/ska/share/eng_archive/update_archive.py $DRY --data-root=$DATAROOT --truncate=$TRUNCATE --content=$CONTENT"
  echo $CMD
  $CMD &>> trunc.log

  CMD="/proj/sot/ska/share/eng_archive/update_archive.py $DRY --data-root=$DATAROOT --content=$CONTENT --max-arch-files=1000"
  echo $CMD
  $CMD &>> trunc.log

done
