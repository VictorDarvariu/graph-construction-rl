#! /bin/bash

echo "<<UPDATING DOCKER IMAGE...>>"
docker pull victordarvariu/rabbitconda:latest
docker build -f $RN_SOURCE_DIR/docker/relnet/Dockerfile -t victordarvariu/relnet --rm $RN_SOURCE_DIR && docker image prune -f

build_targets='mongodb manager worker_cpu worker_gpu'

docker-compose -f $RN_SOURCE_DIR/docker-compose.yml build --force-rm $build_targets  && docker image prune -f
