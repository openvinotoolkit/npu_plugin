#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(BUILD_COMPILER_FOR_DRIVER)
    set(COMPILER_FOR_DRIVER_VERSION_FILE ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_al/src/version.cpp)
endif()

if(NOT ENABLE_NPU_MONO)
    add_definitions(-DIE_BUILD_POSTFIX=${IE_BUILD_POSTFIX_VAR})
endif()

add_subdirectory(vpux_al)

add_subdirectory(vpux_ngraph_transformations)

if(ENABLE_ZEROAPI_BACKEND)
    add_subdirectory(zero_backend)
endif()

if(BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(vpux_driver_compiler)
endif()
add_subdirectory(vpux_plugin)

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    add_subdirectory(vpux_driver_compiler_adapter)
endif()

add_subdirectory(vpux_utils)

add_subdirectory(vpux_compiler)

if(BUILD_SHARED_LIBS AND ENABLE_MLIR_COMPILER)
    add_subdirectory(vpux_translate_utils)
endif()

if(ENABLE_IMD_BACKEND)
    add_subdirectory(vpux_imd_backend)
endif()
