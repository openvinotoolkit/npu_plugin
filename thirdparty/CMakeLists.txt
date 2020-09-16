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

file(MAKE_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/vpu")

#
# vpual
#

if(AARCH64)
    find_library(VPUSMM_LIBRARY vpusmm ONLY_CMAKE_FIND_ROOT_PATH)
    add_library(vpusmm UNKNOWN IMPORTED)
    set_target_properties(vpusmm PROPERTIES
        IMPORTED_LOCATION ${VPUSMM_LIBRARY})

    find_package(vpualHost REQUIRED CONFIG NO_DEFAULT_PATH
        PATHS "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/artifacts/vpualHostInstallPackage/share/vpualHost"
        CMAKE_FIND_ROOT_PATH_BOTH)

    set_target_properties(NN PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(VpualDispatcher PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(sipp_custom PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(XLink PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(RemoteFlic PROPERTIES IMPORTED_GLOBAL TRUE)

    add_custom_target(vpual_copy_library ALL
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_PROPERTY:NN,LOCATION>
            $<TARGET_PROPERTY:VpualDispatcher,LOCATION>
            $<TARGET_PROPERTY:sipp_custom,LOCATION>
            $<TARGET_PROPERTY:XLink,LOCATION>
            $<TARGET_PROPERTY:RemoteFlic,LOCATION>
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/vpu"
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
    set(VPUSMM_LIBRARY "" CACHE INTERNAL "")

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

#
# mcmCompiler
#

set(MCM_COMPILER_BASE_DIR "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/artifacts/mcmCompilerInstallPackage")

if(LINUX AND X86_64)
    find_package(mcmCompiler REQUIRED CONFIG NO_DEFAULT_PATH
        PATHS "${MCM_COMPILER_BASE_DIR}/share/cmake/mcmCompiler"
        CMAKE_FIND_ROOT_PATH_BOTH)

    set_target_properties(cm PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(model PROPERTIES IMPORTED_GLOBAL TRUE)
    set_target_properties(metamodel PROPERTIES IMPORTED_GLOBAL TRUE)

    # TODO: should be done in compiler CMake files
    get_target_property(MCM_INCLUDE_DIR cm INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(cm
        INTERFACE
            ${MCM_INCLUDE_DIR}
            "${MCM_INCLUDE_DIR}/mcm/meta"
            "${MCM_INCLUDE_DIR}/mcm/meta/include")

    add_library(MCM::cm ALIAS cm)

    add_custom_target(mcm_compiler_copy ALL
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_PROPERTY:cm,LOCATION>
            $<TARGET_PROPERTY:model,LOCATION>
            $<TARGET_PROPERTY:metamodel,LOCATION>
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${MCM_COMPILER_BASE_DIR}/config"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mcm_config"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/mcm_config"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mcm_config"
        COMMENT "[MCM Compiler] Copy lib and config files")
endif()

set(FLATBUFFERS_INCLUDE_PATH "${MCM_COMPILER_BASE_DIR}/include" CACHE INTERNAL "")
set(SCHEMA_INCLUDE_PATH "${MCM_COMPILER_BASE_DIR}/include/mcm/meta" CACHE INTERNAL "")

#
# Firmware
#

# copy firmware to lib folder to have vpu.bin in a package
add_custom_target(vpu_bin_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        "${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2/vpu.bin"
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/vpu"
    COMMENT "[Firmware] Copy vpu.bin")
