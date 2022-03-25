#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "kmb_utils")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "precomp.hpp"
)

enable_warnings_as_errors(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
)

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        ${VPUAL_NN_LIBRARY}
        ${VPUMGR_LIBRARY}
        IE::inference_engine_plugin_api
    PUBLIC
        IE::inference_engine_legacy
        vpux_utils
)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME}
        FOR_TARGETS ${TARGET_NAME}
        EXCLUDE_PATTERNS ".*precomp.hpp"
        ALL
    )
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()

if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
endif()
