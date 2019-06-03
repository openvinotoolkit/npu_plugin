# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


add_subdirectory(pugixml)
export(TARGETS pugixml NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")
export(TARGETS pugixml NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
