echo "EXPORTING VARIABLES"
export WORKERS=4

export VERSION_MAJOR=0 
export VERSION_MINOR=0 
export VERSION_PATCH=0
export RELEASE=dev

export PROJECT_PATH=${HOME}/System/Engineering/projects/fot/platform

export ENG_ARCHIVE=${PROJECT_PATH}/development_archive

export TELEMETRY_ARCHIVE=${ENG_ARCHIVE}/archive 
export STAGING_DIRECTORY=${ENG_ARCHIVE}/staging
export ALL_KNOWN_MSID_METAFILE=${TELEMETRY_ARCHIVE}/all_known_msid_sematics.h5

export JETA_LOGS=${ENG_ARCHIVE}/logs
export JETA_SCRIPTS=${PROJECT_PATH}/jeta/scripts
export JETA_ARCHIVE_DEFINITION_SOURCE=${JETA_SCRIPTS}/sql/create.archive.meta.sql  

export NAME_COLUMN="Telemetry Mnemonic"
export TIME_COLUMN="Observatory Time"
export VALUE_COLUMN="EU Value"

export RAVEN_SECRET_KEY=cGlvdXl0aXVmaGpjbWJuajhoOTdweWlsaG45cDc2ODU3aWRyeWljZmtndmpoYms7amk4dTB5OTdneWl2aGtiamxuamk4MDk3Z3lpYmhrCg 

export JUPYTERHUB_HOME=${ENG_ARCHIVE}/jupyterhub 
export JUPYTERHUB_CONFIG_DIRECTORY=${JUPYTERHUB_HOME}/config 
export JUPYTERHUB_LOG_DIRECTORY=${JUPYTERHUB_HOME}/log  
export JUPYTERHUB_USER_SPACE=${JUPYTERHUB_HOME}/user_space 

export NETWORK_SUBNET=192.168.1.0/24
