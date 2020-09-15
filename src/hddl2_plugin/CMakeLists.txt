#
# Copyright 2019-2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

list(FILTER SOURCES EXCLUDE REGEX "hddl2_plugin.cpp")

set(HDDL2_PLUGIN_TARGET "HDDL2Plugin" CACHE INTERNAL "")
set(HDDL2_PLUGIN_OBJS_TARGET "${HDDL2_PLUGIN_TARGET}_objs" CACHE INTERNAL "")
set(HDDL2_PLUGIN_STATIC_TARGET "${HDDL2_PLUGIN_TARGET}_static" CACHE INTERNAL "")
set(HDDL2_PLUGIN_ENGINE_SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/hddl2_plugin.cpp" CACHE INTERNAL "")

#
# Object library
#

add_library(${HDDL2_PLUGIN_OBJS_TARGET} OBJECT ${SOURCES})

enable_warnings_as_errors(${HDDL2_PLUGIN_OBJS_TARGET})
addVersionDefines(${HDDL2_PLUGIN_ENGINE_SOURCE_FILE} CI_BUILD_NUMBER)
target_compile_definitions(${HDDL2_PLUGIN_OBJS_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${HDDL2_PLUGIN_OBJS_TARGET}
    PUBLIC
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/vpu"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/hddl2"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hddl_unite")

link_system_libraries(${HDDL2_PLUGIN_OBJS_TARGET}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
        HddlUnite)
target_link_libraries(${HDDL2_PLUGIN_OBJS_TARGET}
    PUBLIC
        kmb_utils
        vpux)

add_cpplint_target(${HDDL2_PLUGIN_OBJS_TARGET}_cpplint FOR_TARGETS ${HDDL2_PLUGIN_OBJS_TARGET})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${HDDL2_PLUGIN_TARGET} FOR_TARGETS ${HDDL2_PLUGIN_OBJS_TARGET} ALL)
    add_dependencies(${HDDL2_PLUGIN_OBJS_TARGET} clang_format_${HDDL2_PLUGIN_TARGET})
endif()

#
# Static library
#

add_library(${HDDL2_PLUGIN_STATIC_TARGET} STATIC
    $<TARGET_OBJECTS:${HDDL2_PLUGIN_OBJS_TARGET}>
    ${HDDL2_PLUGIN_ENGINE_SOURCE_FILE})

target_compile_definitions(${HDDL2_PLUGIN_STATIC_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${VPUX_DEVICE_NAME}\"")

target_include_directories(${HDDL2_PLUGIN_STATIC_TARGET}
    PUBLIC
        $<TARGET_PROPERTY:${HDDL2_PLUGIN_OBJS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)

link_system_libraries(${HDDL2_PLUGIN_STATIC_TARGET}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
    PRIVATE
        HddlUnite)
target_link_libraries(${HDDL2_PLUGIN_STATIC_TARGET}
    PUBLIC
        kmb_utils
    PRIVATE
        vpux)
