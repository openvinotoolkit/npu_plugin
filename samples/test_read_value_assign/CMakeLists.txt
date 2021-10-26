# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME read_value_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS detection_sample.h
              DEPENDENCIES IE::format_reader ${VPUMGR_LIBRARY} kmb_utils IE::ie_samples_utils)
