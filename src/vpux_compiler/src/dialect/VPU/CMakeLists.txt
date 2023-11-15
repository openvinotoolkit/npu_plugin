#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
set(TARGET_NAME "npu_compiler_dialect_vpu")
add_mlir_dialect_library(${TARGET_NAME}
    STATIC
        ${SRC_FILES}
        # FIXME: Need a solution to replace the function (E#88484)
        ${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/dialect/IE/ops/sparsity_statistics.cpp
    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    LINK_LIBS PRIVATE
        npu_utils
        inferenceStatic
    DEPENDS
        MLIRVPUXIncGenList)
target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        ${VPUNN_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}
        ${VPU_COMPILER_BIN_INCLUDE_DIR})

if(TARGET IE::nlohmann_json)
    target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<TARGET_PROPERTY:IE::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
else()
    find_package(nlohmann_json 3.9.0 QUIET)
    if(nlohmann_json_FOUND)
        target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
            $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
    else()
        message(FATAL_ERROR "Failed to find system nlohmann_json.")
    endif()
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR})
add_src_target($<BUILD_INTERFACE:${TARGET_NAME}>)
vpux_enable_clang_format(${TARGET_NAME})
enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
