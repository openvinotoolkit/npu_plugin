#
# Copyright 2019 Intel Corporation.
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

set(TARGET_NAME "kmbPlugin")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
endif()

set(EXTRA_COMPILE_OPTIONS -Wall -Wextra -Werror)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR kmb_plugin.cpp
              DEVICE_NAME KMB)

target_compile_options(${TARGET_NAME} PRIVATE ${EXTRA_COMPILE_OPTIONS})

# TODO: Remove when warnings are fixed in vpualHost
set_source_files_properties(kmb_preproc_gapi_kernels_sipp.cpp PROPERTIES
                            COMPILE_FLAGS "-Wno-error")

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        ${VPUSMM_INCLUDES}
        ${VPUAL_NN_INCLUDES}
        ${GAPI_SIPP_INCLUDES}
    PUBLIC
        "${IE_FRONTEND_MCM_INCLUDE_DIR}"
    PRIVATE
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_NAME}
    PUBLIC
        IE::inference_engine
        IE::vpu_common_lib
        frontend_mcm
        ${VPUAL_NN_LIBRARY}
        ${VPUSMM_LIBRARY}
    PRIVATE
        ${GAPI_SIPP_LIBRARY}
        kmb_utils)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
endif()

if(ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    target_compile_options(${TARGET_NAME}_test_static PRIVATE ${EXTRA_COMPILE_OPTIONS})

    target_include_directories(${TARGET_NAME}_test_static
        SYSTEM PRIVATE
            ${VPUSMM_INCLUDES}
            ${VPUAL_NN_INCLUDES}
            ${GAPI_SIPP_INCLUDES}
            "${IE_FRONTEND_MCM_INCLUDE_DIR}"
        PUBLIC
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
            ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET_NAME}_test_static
        PUBLIC
            IE::inference_engine
            IE::vpu_common_lib
            frontend_mcm
            ${VPUAL_NN_LIBRARY}
            ${VPUSMM_LIBRARY}
        PRIVATE
            ${GAPI_SIPP_LIBRARY}
            kmb_utils)
endif()
