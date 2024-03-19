#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

find_package(ZLIB)

set(TARGET_NAME npu_algo_utils)

list(APPEND SOURCES
                simulated_annealing.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})
vpux_enable_clang_format(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/
        ${VPUX_UTILS_CURRENT_SOURCE_DIR}/include/vpux/utils/algorithm
)

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()

add_dependencies(${TARGET_NAME}
    npu_llvm_utils
)

target_link_libraries(${TARGET_NAME} PRIVATE npu_llvm_utils)
