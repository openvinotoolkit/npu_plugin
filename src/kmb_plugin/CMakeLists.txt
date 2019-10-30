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

set(TARGET_NAME "kmbPlugin")

if(ENABLE_VPUAL)
    set(VPUAL_NN_LIBRARY NN)
    set(GAPI_SIPP_LIBRARY gapi_sipp)
endif()
if(ENABLE_MCM_COMPILER)
    set(MCM_COMPILER_LIBRARY cm)
endif()

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

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
    PRIVATE
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        )

target_link_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        vpusmm
        IE::vpu_common_lib
        kmb_utils
        ${VPUAL_NN_LIBRARY}
        ${GAPI_SIPP_LIBRARY}
        ${MCM_COMPILER_LIBRARY})

if(ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    target_compile_options(${TARGET_NAME}_test_static PRIVATE ${EXTRA_COMPILE_OPTIONS})

    target_include_directories(${TARGET_NAME}_test_static
        PUBLIC
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            )
    target_link_libraries(${TARGET_NAME}_test_static
        PUBLIC
            IE::inference_engine
            IE::inference_engine_plugin_api
            vpusmm
        PRIVATE
            IE::vpu_common_lib
            kmb_utils
            ${VPUAL_NN_LIBRARY}
            ${GAPI_SIPP_LIBRARY}
            ${MCM_COMPILER_LIBRARY})
endif()
