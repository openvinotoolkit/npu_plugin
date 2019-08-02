# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME kmb_infer_sample
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS infer_app.h
              DEPENDENCIES format_reader)
