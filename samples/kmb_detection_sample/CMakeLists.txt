# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(AARCH64)
ie_add_sample(NAME kmb_detection_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS detection_sample.h
              DEPENDENCIES IE::format_reader ${VPUMGR_LIBRARY} kmb_utils IE::ie_samples_utils)
endif()
