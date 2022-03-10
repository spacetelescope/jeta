#!/bin/bash

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
# THESE VALUES ARE OVERWRITTEN BY THE CI/CD
# PIPELINE DURING A PRODUCTION DEPLOYMENT
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
set -x \
&& echo "EXPORTING VARIABLES"

# The number of workers that can be spaw
export WORKERS=4

# CLI Version Information
export VERSION_MAJOR=0 
export VERSION_MINOR=0 
export VERSION_PATCH=0
export RELEASE=dev

# Project Root Directory
export PROJECT_PATH=${HOME}/Projects/repos/jeta

# Archive Root Directory
export ENG_ARCHIVE=${HOME}/local_archive

# Archive Data Root Directory (i.e. telemetery and archive metadata goes here)
export TELEMETRY_ARCHIVE=${ENG_ARCHIVE}/archive 

# Staging Area Root Directory (i.e. Where files are delivered to be processed)
export STAGING_DIRECTORY=${ENG_ARCHIVE}/staging

# MSID Data Reference File
export ALL_KNOWN_MSID_METAFILE=${ENG_ARCHIVE}/all_known_msid_sematics.h5

# Root Directory for System Logs
export JETA_LOGS=${ENG_ARCHIVE}/logs

# Root Directory for System Support Scripts
export JETA_SCRIPTS=${PROJECT_PATH}/scripts

# Path to Meta Database Definition File
export JETA_ARCHIVE_DEFINITION_SOURCE=${JETA_SCRIPTS}/sql/create.archive.meta.sql  

# Relevant Field Names
export NAME_COLUMN="Telemetry Mnemonic"
export TIME_COLUMN="Observatory Time"
export VALUE_COLUMN="EU Value"

# Dummy Web API Session Key 
export RAVEN_SECRET_KEY=cGlvdXl0aXVmaGpjbWJuajhoOTdweWlsaG45cDc2ODU3aWRyeWljZmtndmpoYms7amk4dTB5OTdneWl2aGtiamxuamk4MDk3Z3lpYmhrCg 

# JupyterLab Specific Variables
export JUPYTERHUB_HOME=${ENG_ARCHIVE}/jupyterhub 
export JUPYTERHUB_CONFIG_DIRECTORY=${JUPYTERHUB_HOME}/config 
export JUPYTERHUB_LOG_DIRECTORY=${JUPYTERHUB_HOME}/log  
export JUPYTERHUB_USER_SPACE=${JUPYTERHUB_HOME}/user_space 

# Local Network Subnet for Docker
export NETWORK_SUBNET=192.168.1.0/24
