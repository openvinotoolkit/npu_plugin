#!/bin/bash

rm -rf ./mvbuild
echo Copying yolo_v2_tiny_custom.blob to ./blobs folder
mkdir -p blobs
cp tests/yolo_v2_tiny_custom/yolo_v2_tiny_custom.blob blobs/

TESTS=(yolo_v2_tiny_custom)

. $PWD/common/test_executor.sh

python ./tests/yolo_v2_tiny_custom/parse-yolo.py ./tests/yolo_v2_tiny_custom/out0.bin
