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

list(FILTER SOURCES EXCLUDE REGEX "kmb_plugin.cpp")

set(KMB_PLUGIN_TARGET "kmbPlugin" CACHE INTERNAL "")
set(KMB_PLUGIN_OBJS_TARGET "${KMB_PLUGIN_TARGET}_objs" CACHE INTERNAL "")
set(KMB_PLUGIN_STATIC_TARGET "${KMB_PLUGIN_TARGET}_static" CACHE INTERNAL "")
set(KMB_PLUGIN_ENGINE_SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/kmb_plugin.cpp" CACHE INTERNAL "")

#
# Object library
#

add_library(${KMB_PLUGIN_OBJS_TARGET} OBJECT ${SOURCES})

enable_warnings_as_errors(${KMB_PLUGIN_OBJS_TARGET})

target_compile_definitions(${KMB_PLUGIN_OBJS_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${KMB_PLUGIN_OBJS_TARGET}
    PUBLIC
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include/vpu"
        ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${KMB_PLUGIN_OBJS_TARGET}
    SYSTEM PRIVATE
        ${VPUAL_NN_INCLUDES}
        ${GAPI_SIPP_INCLUDES}
        ${GAPI_M2I_INCLUDES})

link_system_libraries(${KMB_PLUGIN_OBJS_TARGET}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
        ${VPUAL_NN_LIBRARY}
        ${GAPI_SIPP_LIBRARY}
        ${GAPI_M2I_LIBRARY}
        ${VPUSMM_LIBRARY})
target_link_libraries(${KMB_PLUGIN_OBJS_TARGET}
    PUBLIC
        kmb_utils
        vpux)

add_cpplint_target(${KMB_PLUGIN_OBJS_TARGET}_cpplint FOR_TARGETS ${KMB_PLUGIN_OBJS_TARGET})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${KMB_PLUGIN_TARGET} FOR_TARGETS ${KMB_PLUGIN_OBJS_TARGET} ALL)
    add_dependencies(${KMB_PLUGIN_OBJS_TARGET} clang_format_${KMB_PLUGIN_TARGET})
endif()

#
# Plugin library
#

set(TARGET_NAME ${KMB_PLUGIN_TARGET})
ie_add_plugin(NAME ${KMB_PLUGIN_TARGET}
    DEVICE_NAME ${KMB_DEVICE_NAME}
    SOURCES
        $<TARGET_OBJECTS:${KMB_PLUGIN_OBJS_TARGET}>
        ${KMB_PLUGIN_ENGINE_SOURCE_FILE}
    VERSION_DEFINES_FOR ${KMB_PLUGIN_ENGINE_SOURCE_FILE})

target_compile_definitions(${KMB_PLUGIN_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${KMB_DEVICE_NAME}\"")

target_link_libraries(${KMB_PLUGIN_TARGET}
    PRIVATE
        ${KMB_PLUGIN_OBJS_TARGET})

install(TARGETS ${KMB_PLUGIN_TARGET}
    RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${KMB_DEVICE_NAME}
    ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${KMB_DEVICE_NAME}
    LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${KMB_DEVICE_NAME})

#
# Static library
#

add_library(${KMB_PLUGIN_STATIC_TARGET} STATIC
    $<TARGET_OBJECTS:${KMB_PLUGIN_OBJS_TARGET}>
    ${KMB_PLUGIN_ENGINE_SOURCE_FILE})

target_compile_definitions(${KMB_PLUGIN_STATIC_TARGET}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        "DEVICE_NAME=\"${KMB_DEVICE_NAME}\"")

target_include_directories(${KMB_PLUGIN_STATIC_TARGET}
    PUBLIC
        $<TARGET_PROPERTY:${KMB_PLUGIN_OBJS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)

link_system_libraries(${KMB_PLUGIN_STATIC_TARGET}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_plugin_api
        IE::vpu_common_lib
        ${VPUAL_NN_LIBRARY}
        ${VPUSMM_LIBRARY}
    PRIVATE
        ${GAPI_SIPP_LIBRARY}
        ${GAPI_M2I_LIBRARY})
target_link_libraries(${KMB_PLUGIN_STATIC_TARGET}
    PUBLIC
        kmb_utils
    PRIVATE
        vpux)
