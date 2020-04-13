#!/bin/bash

rm -rf ./blobs ./blobs_info ./result.csv
rm -rf ./mvbuild

TEST_DIR=${PWD}
BUILD_DIR="${PWD}/../../../build"

export PATH=$PATH:${BUILD_DIR}/contrib/flatbuffers

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR} || { echo "Failure"; exit 1; }
cmake -DCMAKE_BUILD_TYPE=Release ..
make cm -j8
make gen_blobs
cd ${TEST_DIR} || { echo "Failure"; exit 1; }

TESTS=(
        custom_region_chw
        custom_region_hwc
        custom_cvtu8f16
        custom_grn
        custom_reorg_chw
        custom_reorg_hwc
        custom_correlate
        custom_fake_quantize
        custom_resample_noAA
        custom_resample_AA
        custom_ctc_decoder
        custom_st
        custom_fake_binarization
        regionyolo
        reorgyolo
      )

. $PWD/common/test_executor.sh

python3 parse_logs.py | tee result.csv
