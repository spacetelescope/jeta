#!/bin/bash

rm ${TELEMETRY_ARCHIVE}/archive.meta.info.db3 \
&& rm ${TELEMETRY_ARCHIVE}/msids.pickle \
&& rm -rf ${TELEMETRY_ARCHIVE}/data;
