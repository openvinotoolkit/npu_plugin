# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME kmb_multi_classification_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS multi_classification_sample.h
              DEPENDENCIES IE::format_reader kmb_utils IE::ie_samples_utils)
