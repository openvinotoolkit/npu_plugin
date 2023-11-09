#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

list(FILTER SOURCES EXCLUDE REGEX "vpux_plugin.cpp")

set(VPUX_PLUGIN_TARGET "openvino_intel_npu_plugin" CACHE INTERNAL "")
set(VPUX_PLUGIN_STATIC_TARGET "${VPUX_PLUGIN_TARGET}_static" CACHE INTERNAL "")
set(VPUX_PLUGIN_ENGINE_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/vpux_plugin.cpp CACHE INTERNAL "")

addVersionDefines(${VPUX_PLUGIN_ENGINE_SOURCE_FILE} CI_BUILD_NUMBER)
#
# Object library
#
if(BUILD_SHARED_LIBS)
    set(VPUX_PLUGIN_OBJS_TARGET ${VPUX_PLUGIN_TARGET}_objs)
    add_library(${VPUX_PLUGIN_OBJS_TARGET} OBJECT ${SOURCES})
endif()

#
# Plugin library
#

if(NOT BUILD_SHARED_LIBS)
    set(VPUX_PLUGIN_TARGET_SOURCES ${SOURCES})
endif()

ov_add_plugin(NAME ${VPUX_PLUGIN_TARGET}
    DEVICE_NAME ${NPU_DEVICE_NAME}
    SOURCES
        ${VPUX_PLUGIN_TARGET_SOURCES}
        ${VPUX_PLUGIN_ENGINE_SOURCE_FILE}
    OBJECT_LIBRARIES
        ${VPUX_PLUGIN_OBJS_TARGET}
    VERSION_DEFINES_FOR ${VPUX_PLUGIN_ENGINE_SOURCE_FILE})

if(BUILD_SHARED_LIBS)
    set(TARGET_NAME ${VPUX_PLUGIN_OBJS_TARGET} CACHE INTERNAL "")
    target_link_libraries(${VPUX_PLUGIN_TARGET} PUBLIC ${VPUX_PLUGIN_OBJS_TARGET})
else()
    set(TARGET_NAME ${VPUX_PLUGIN_TARGET} CACHE INTERNAL "")
    if(ENABLE_ZEROAPI_BACKEND)
        target_link_libraries(${VPUX_PLUGIN_TARGET} INTERFACE npu_level_zero_backend)
    endif()
    target_compile_definitions(${VPUX_PLUGIN_TARGET} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

enable_warnings_as_errors(${TARGET_NAME})
replace_noerror(${TARGET_NAME})

vpux_enable_clang_format(${TARGET_NAME})
target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

link_system_libraries(${TARGET_NAME}
    PUBLIC
        openvino::runtime
        openvino::runtime::dev
        npu_utils
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        kmb_utils
        npu_al
    PRIVATE
        npu_utils
)

#
# Static library
#

if(BUILD_SHARED_LIBS)
    add_library(${VPUX_PLUGIN_STATIC_TARGET} STATIC
            ${VPUX_PLUGIN_ENGINE_SOURCE_FILE}
    )
    target_link_libraries(${VPUX_PLUGIN_STATIC_TARGET}
        PUBLIC
            ${VPUX_PLUGIN_OBJS_TARGET}
    )
else()
    add_library(${VPUX_PLUGIN_STATIC_TARGET} ALIAS ${VPUX_PLUGIN_TARGET})
endif()

set(ALL_PLUGINS)

list(APPEND ALL_PLUGINS ${VPUX_PLUGIN_TARGET})

ie_register_plugins(MAIN_TARGET ${VPUX_PLUGIN_TARGET} POSSIBLE_PLUGINS ${ALL_PLUGINS})
if(WIN32)
    ie_add_api_validator_post_build_step(TARGET ${VPUX_PLUGIN_TARGET})
endif()
