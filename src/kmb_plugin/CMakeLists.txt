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

if (ENABLE_VPUAL)
    add_definitions(-DENABLE_VPUAL)
    set(VPUAL_NN_LIBRARY NN)
    set(GAPI_SIPP_LIBRARY gapi_sipp)
endif()

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(NAME ${TARGET_NAME}
              SOURCES ${SOURCES}
              VERSION_DEFINES_FOR kmb_plugin.cpp
              DEVICE_NAME KMB)

target_compile_options(${TARGET_NAME}
        PRIVATE
        -Wall
        -Wextra
        -Werror)

# TODO: Remove when warnings are fixed in vpualHost
set_source_files_properties(kmb_preproc_gapi_kernels_sipp.cpp PROPERTIES
                            COMPILE_FLAGS "-Wno-error")

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS ${TARGET_NAME})

if (ENABLE_MCM_COMPILER)
    add_dependencies(${TARGET_NAME} mcmCompiler)
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        )

target_include_directories(${TARGET_NAME}
    SYSTEM PRIVATE
        "${MCM_COMPILER_INCLUDE_PATH}"
        "${FLATBUFFERRS_INCLUDE_PATH}"
        "${MCM_SCHEMA_INCLUDE_PATH}"
        "${METIS_DIR}/include"
        )

target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${INTEL_ITT_LIBS}
        IE::inference_engine
        IE::vpu_common_lib
        ${MCM_COMPILER_LIBRARY}
        ${VPUAL_NN_LIBRARY}
        vpusmm
        ${GAPI_SIPP_LIBRARY}
        kmb_utils
        ${METIS_DIR}
        )

if(ENABLE_TESTS)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})
    if (ENABLE_MCM_COMPILER)
        add_dependencies(${TARGET_NAME}_test_static mcmCompiler)
    endif()

    target_compile_options(${TARGET_NAME}_test_static
            PRIVATE
            -Wall
            -Werror)

    target_include_directories(${TARGET_NAME}_test_static
        PUBLIC
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}"
            )
    target_include_directories(${TARGET_NAME}_test_static
        SYSTEM PRIVATE
            "${MCM_COMPILER_INCLUDE_PATH}"
            "${FLATBUFFERRS_INCLUDE_PATH}"
            "${MCM_SCHEMA_INCLUDE_PATH}"
            "${METIS_DIR}/include"
            )
    target_link_libraries(${TARGET_NAME}_test_static
            PUBLIC
                vpusmm
                IE::inference_engine
                IE::inference_engine_plugin_api
            PRIVATE
                ${INTEL_ITT_LIBS}
                IE::vpu_common_lib
                ${MCM_COMPILER_LIBRARY}
                ${VPUAL_NN_LIBRARY}
                ${GAPI_SIPP_LIBRARY}
                kmb_utils)
endif()

if(ENABLE_MCM_COMPILER)
    # copy mcm compiler library to target directory when the compiler is enabled
    set(BINARY_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")
    add_custom_target(mcm_copy_libraries ALL
        COMMAND "${CMAKE_COMMAND}" -E copy "${MCM_COMPILER_LIBRARY}" "${BINARY_OUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${MCM_COMPILER_MODEL_LIBRARY}" "${BINARY_OUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${MCM_COMPILER_METAMODEL_LIBRARY}" "${BINARY_OUT_DIR}"
        COMMENT "[MCM_COMPILER] Copy mcm compiler libraries to build results")
endif()
