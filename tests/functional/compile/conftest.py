#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

""" Configuration for models compilation tests.

Models comilation tests run compilation for models and pass if compilation
succeded without errors.

Pre-requisite to run the tests are model packages.

Usage:
python3 -m pytest  --html=compile.html --models=<path to model packages> --compiler=vpu2_compile test_compile.py
"""

import tempfile
from types import SimpleNamespace

# Keem Bay KPI models list
# find <models path> -wholename "*/FP16-INT8/*.xml"
KMB_KPI_MODELS = [
    'vd_kmb_models_intel_ww22.tar.bz2/yolo-v2-ava-0001/tf/FP16-INT8/yolo-v2-ava-0001.xml',
    'vd_kmb_models_intel_ww22.tar.bz2/yolo-v2-ava-sparse-35-0001/tf/FP16-INT8/yolo-v2-ava-sparse-35-0001.xml',
    'vd_kmb_models_intel_ww22.tar.bz2/yolo-v2-ava-sparse-70-0001/tf/FP16-INT8/yolo-v2-ava-sparse-70-0001.xml',
    'vd_kmb_models_intel_ww22.tar.bz2/yolo-v2-tiny-ava-0001/tf/FP16-INT8/yolo-v2-tiny-ava-0001.xml',
    'vd_kmb_models_intel_ww22.tar.bz2/yolo-v2-tiny-ava-sparse-30-0001/tf/FP16-INT8/yolo-v2-tiny-ava-sparse-30-0001.xml',
    'vd_kmb_models_intel_ww22.tar.bz2/yolo-v2-tiny-ava-sparse-60-0001/tf/FP16-INT8/yolo-v2-tiny-ava-sparse-60-0001.xml',
    'vd_kmb_models_public_ww22.tar.bz2/resnet-50-pytorch/caffe2/FP16-INT8/resnet-50-pytorch.xml',
    'vd_kmb_models_public_ww22.tar.bz2/mobilenet-v2/caffe2/FP16-INT8/mobilenet-v2.xml',
    'vd_kmb_models_public_ww22.tar.bz2/googlenet-v1/tf/FP16-INT8/googlenet-v1.xml',
    'vd_kmb_models_public_ww22.tar.bz2/googlenet-v3/tf/FP16-INT8/googlenet-v3.xml',
    'vd_kmb_models_public_ww22.tar.bz2/squeezenet1.1/caffe2/FP16-INT8/squeezenet1.1.xml',

    'vd_kmb_models_intel_ww22.tar.bz2/faster-rcnn-resnet101-coco-sparse-60-0001/tf/FP16-INT8/faster-rcnn-resnet101-coco-sparse-60-0001.xml',
    'vd_kmb_models_public_ww22.tar.bz2/ssd512/caffe/FP16-INT8/ssd512.xml',
    'vd_kmb_models_intel_ww22.tar.bz2/icnet-camvid-ava-0001/tf/FP16-INT8/icnet-camvid-ava-0001.xml',
]


def pytest_addoption(parser):
    """ Define extra options for pytest options
    """
    parser.addoption('--models', help='Models packages')
    parser.addoption('--compiler', help='Model compiler tool')
    parser.addoption('--output', help='Output durectory for compiled models')


def pytest_generate_tests(metafunc):
    """ Generate tests depending on command line options
    """
    out = metafunc.config.getoption("output")
    if not out:
        out = tempfile.mkdtemp(prefix='_blobs-', dir='.')
    metafunc.parametrize("param_ir", [SimpleNamespace(model=net,
                                                      compiler_tool=metafunc.config.getoption(
                                                          "compiler"),
                                                      models_dir=metafunc.config.getoption(
                                                          "models"),
                                                      output_dir=out) for net in KMB_KPI_MODELS],
                         ids=KMB_KPI_MODELS)
