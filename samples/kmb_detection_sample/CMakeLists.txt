# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME kmb_detection_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/region_yolov2tiny.cpp"
              HEADERS detection_sample.h
              DEPENDENCIES format_reader vpusmm)
