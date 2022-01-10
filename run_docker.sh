#!/bin/sh

export DISPLAY=$DISPLAY

# set -x \
# && conda activate jeta_standalone

docker-compose down && docker-compose build
docker-compose up