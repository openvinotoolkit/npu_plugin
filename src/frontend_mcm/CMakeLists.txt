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

add_library(${TARGET_NAME} STATIC ${SOURCES})

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
    SYSTEM PRIVATE
        ${MCM_COMPILER_INCLUDES}
        ${MCM_COMPILER_INCLUDES}/include
        ${META_DIR_PATH}
        "${FLATBUFFERS_INCLUDE_PATH}"
        "${SCHEMA_INCLUDE_PATH}"
    PUBLIC
        "include"
        ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        IE::inference_engine
        IE::inference_engine_transformations
        IE::inference_engine_lp_transformations
        IE::vpu_common_lib
        vpux
    PRIVATE
        IE::pugixml
        kmb_utils
        ${MCM_COMPILER_LIBRARY})

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
