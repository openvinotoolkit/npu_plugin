# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME kmb_full_pipeline_compile_app
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS classification_sample_async.h
              DEPENDENCIES format_reader)