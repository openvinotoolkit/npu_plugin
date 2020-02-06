# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME kmbFuncTests)

file(GLOB_RECURSE TEST_INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

file(GLOB_RECURSE TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

# Create named folders for the sources within the .vcproj
source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE})

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_link_libraries(${TARGET_NAME} PRIVATE
        IE::funcSharedTests
        )
