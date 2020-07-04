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

function(add_3rdparty_subdir SUBDIR_NAME)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${SUBDIR_NAME}")
        message(FATAL_ERROR "${SUBDIR_NAME} was not found. Please run `git submodule update --init --recursive`")
    endif()

    # Remove problematic compiler flags
    foreach(flag "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
        string(REPLACE ${flag} "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
        string(REPLACE ${flag} "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    endforeach()

    set(CMAKE_C_VISIBILITY_PRESET default)
    set(CMAKE_CXX_VISIBILITY_PRESET default)

    add_subdirectory(${SUBDIR_NAME})
endfunction()

if(ARM OR AARCH64)
    add_custom_target(vpual_copy_library ALL
            COMMAND "${CMAKE_COMMAND}" -E copy  $<TARGET_PROPERTY:NN,LOCATION>
                                                $<TARGET_PROPERTY:VpualDispatcher,LOCATION>
                                                $<TARGET_PROPERTY:sipp_custom,LOCATION>
                                                $<TARGET_PROPERTY:XLink,LOCATION>
                                                $<TARGET_PROPERTY:RemoteFlic,LOCATION>
            $<TARGET_FILE_DIR:kmbPlugin>
            COMMENT "[VPUAL] Copy vpual libraries to build results")

    add_subdirectory(gapi_sipp/sipp)
    add_subdirectory(gapi_sipp/m2i)

    if(ENABLE_TESTS)
        add_subdirectory(tests/integration)
    endif()

    set(VPUAL_XLINK_LIBRARY XLink CACHE INTERNAL "")
    set(VPUAL_XLINK_INCLUDES $<TARGET_PROPERTY:XLink,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUAL_DISPATCHER_LIBRARY VpualDispatcher CACHE INTERNAL "")
    set(VPUAL_DISPATCHER_INCLUDES $<TARGET_PROPERTY:VpualDispatcher,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(VPUAL_NN_LIBRARY NN CACHE INTERNAL "")
    set(VPUAL_NN_INCLUDES $<TARGET_PROPERTY:NN,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(GAPI_SIPP_LIBRARY gapi_sipp CACHE INTERNAL "")
    set(GAPI_SIPP_INCLUDES $<TARGET_PROPERTY:gapi_sipp,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    set(GAPI_M2I_LIBRARY gapi_m2i CACHE INTERNAL "")
    set(GAPI_M2I_INCLUDES $<TARGET_PROPERTY:gapi_m2i,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")
else()
    set(VPUAL_XLINK_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_XLINK_INCLUDES "" CACHE INTERNAL "")

    set(VPUAL_DISPATCHER_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_DISPATCHER_INCLUDES "" CACHE INTERNAL "")

    set(VPUAL_NN_LIBRARY "" CACHE INTERNAL "")
    set(VPUAL_NN_INCLUDES "" CACHE INTERNAL "")

    set(GAPI_SIPP_LIBRARY "" CACHE INTERNAL "")
    set(GAPI_SIPP_INCLUDES "" CACHE INTERNAL "")

    set(GAPI_M2I_LIBRARY "" CACHE INTERNAL "")
    set(GAPI_M2I_INCLUDES "" CACHE INTERNAL "")
endif()

if(mcmCompiler_FOUND)
    set(COMPILATION_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")

    add_custom_target(mcm_compiler_copy_libs ALL
            COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_PROPERTY:cm,LOCATION>
                                               $<TARGET_PROPERTY:model,LOCATION>
                                               $<TARGET_PROPERTY:metamodel,LOCATION>
                                               ${COMPILATION_OUT_DIR}
            COMMENT "[MCM Compiler] Copy lib files")

    set(mcmCompilerDescriptor_DIR "${mcmCompiler_DIR}/../../../config")

    add_custom_target(mcm_compiler_copy_configs ALL
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${mcmCompilerDescriptor_DIR}"
                                                         "${COMPILATION_OUT_DIR}/mcm_config"
            COMMENT "[MCM Compiler] Copy config files for compilation")

    set(MCM_COMPILER_LIBRARY cm CACHE INTERNAL "")
    set(MCM_COMPILER_INCLUDES $<TARGET_PROPERTY:cm,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "")

    get_target_property(META_DIR_PATH cm INTERFACE_INCLUDE_DIRECTORIES)
    set(META_DIR_PATH "${META_DIR_PATH}/mcm/meta" CACHE INTERNAL "")
else()
    set(MCM_COMPILER_LIBRARY "" CACHE INTERNAL "")
    set(MCM_COMPILER_INCLUDES "" CACHE INTERNAL "")
    set(META_DIR_PATH "" CACHE INTERNAL "")
    # mcmCompiler_DIR is assumed to be pointing at share/cmake/mcmCompiler
    set(FLATBUFFERS_INCLUDE_PATH "${mcmCompiler_DIR}/../../../include/" CACHE INTERNAL "")
    set(SCHEMA_INCLUDE_PATH "${FLATBUFFERS_INCLUDE_PATH}/mcm/meta" CACHE INTERNAL "")
endif()

# copy firmware to lib folder to have vpu.bin in a package
add_custom_target(vpu_bin_copy ALL
        COMMAND "${CMAKE_COMMAND}" -E copy "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/vpu.bin"
                                            $<TARGET_FILE_DIR:kmbPlugin>
        COMMENT "[Firmware] Copy vpu.bin")
