# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hddl2_classification_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS classification_sample.h
              DEPENDENCIES HddlUnite IE::format_reader kmb_utils IE::ie_samples_utils)
