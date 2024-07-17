#!/bin/bash
DOCKER_IMAGE_NAME=ssh-mapping
DOCKERFILE=`pwd`
docker build -t $DOCKER_IMAGE_NAME -f $DOCKERFILE/Dockerfile .