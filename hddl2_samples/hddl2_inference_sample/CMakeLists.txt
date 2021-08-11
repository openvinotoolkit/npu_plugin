# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hddl2_inference_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS inference_sample.h
              DEPENDENCIES IE::format_reader HddlUnite kmb_utils IE::ie_samples_utils)

