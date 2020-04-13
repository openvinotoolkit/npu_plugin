# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieModelsHelpersKmb)

file(GLOB HELPERS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB HELPERS_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/*.hpp
        )

message("configuring file: ${CMAKE_CURRENT_BINARY_DIR}/get_model_repos.hpp")
configure_file(get_model_repos.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/get_model_repos.hpp @ONLY)

add_library(${TARGET_NAME} STATIC ${HELPERS_SRC})

target_link_libraries(${TARGET_NAME} PUBLIC IE::ieTestHelpers)

target_include_directories(${TARGET_NAME}
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}"
        PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}")
