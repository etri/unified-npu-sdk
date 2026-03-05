#!/bin/bash
set -e

# =============================
# unified-sdk build script
# =============================

IMAGE_NAME="unified-sdk"
TAG="cuda12.3"
BASE_IMAGE="nvcr.io/nvidia/tensorrt:23.10-py3"
UID_VALUE=$(id -u)
GID_VALUE=$(id -g)

echo " Building Docker image: ${IMAGE_NAME}:${TAG}"
echo " Base image: ${BASE_IMAGE}"
echo " UID:GID = ${UID_VALUE}:${GID_VALUE}"

# Build (no cache option if needed)
docker build \
  -t ${IMAGE_NAME}:${TAG} \
  --build-arg BASE_IMAGE=${BASE_IMAGE} \
  --build-arg UID=${UID_VALUE} \
  --build-arg GID=${GID_VALUE} \
  .

echo " Build complete!"
echo " Run container with:"
echo " docker run --gpus all -it --security-opt seccomp=unconfined --name ${IMAGE_NAME}_dev -v /home/etri/users/rskim/uDC:/workspace ${IMAGE_NAME}:${TAG}"

