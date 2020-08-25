#
# Copyright 2020 Intel Corporation.
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

set(TARGET_NAME "frontend_mcm")
set(IE_FRONTEND_MCM_INCLUDE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "IE_FRONTEND_MCM_INCLUDE_DIR")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
endif()

add_library(${TARGET_NAME} SHARED ${SOURCES})
target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

if(UNIX)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wall
            -Werror
            -Wextra)
else()
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /wd4146     # unary minus operator applied to unsigned type, result still unsigned
            /wd4996)    # code uses a function, class member, variable, or typedef that's marked deprecated
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${FRONTEND_VPUX_INCLUDE_DIR}
    PUBLIC
        ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
        "include")

# TODO: remove when compiler works on ARM and Windows
if(ARM OR AARCH64 OR WIN32)
    target_include_directories(${TARGET_NAME}
        PRIVATE 
            ${FLATBUFFERS_INCLUDE_PATH}
            ${SCHEMA_INCLUDE_PATH})
endif()

target_link_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_transformations
        IE::inference_engine_lp_transformations
        vpux
        IE::pugixml
        vpux
        IE::ngraph_reference
        kmb_utils
        IE::vpu_common_lib)

# TODO: move to the targer_link_libraries above when compiler works on ARM
# TODO remove NOT WIN32 when we have compiler for Windows
if((X86 OR X86_64) AND NOT WIN32)
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            MCM::cm)
   # FIXME: remove define once we hide all logic for compiler in a separate library
   target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_MCM_COMPILER)
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
