# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

ie_add_sample(NAME kmb_infer_sample
              SOURCES ${SRC}
              HEADERS infer_app.h
              DEPENDENCIES IE::format_reader)
