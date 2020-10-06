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

# FIXME ENABLE_LTO leads to segmentation fault after 'calling fini libfrontend_mcm.so'
# if(ENABLE_LTO)
#     ie_enable_lto()
# endif()

set(TARGET_NAME "frontend_mcm")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} SHARED ${SOURCES})

enable_warnings_as_errors(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

if(MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /wd4146     # unary minus operator applied to unsigned type, result still unsigned
            /wd4996)    # code uses a function, class member, variable, or typedef that's marked deprecated
endif()

target_include_directories(${TARGET_NAME}
    PUBLIC
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include")

# TODO: remove when compiler works on ARM and Windows
if(AARCH64 OR WIN32)
    target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
            ${FLATBUFFERS_INCLUDE_PATH}
            ${SCHEMA_INCLUDE_PATH})
endif()

link_system_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::inference_engine_transformations
        IE::pugixml
        IE::ngraph_reference
        IE::vpu_common_lib)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        vpux
        kmb_utils)

# TODO: move to the targer_link_libraries above when compiler works on ARM
# TODO remove NOT WIN32 when we have compiler for Windows
if(X86_64 AND NOT WIN32)
    link_system_libraries(${TARGET_NAME}
        PRIVATE
            MCM::cm)

   # FIXME: remove define once we hide all logic for compiler in a separate library
   target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_MCM_COMPILER)
endif()

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
