# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME kmb_detection_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/region_yolov2tiny.cpp"
              HEADERS detection_sample.h
              DEPENDENCIES IE::format_reader vpusmm kmb_utils)
