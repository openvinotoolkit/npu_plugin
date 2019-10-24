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

add_subdirectory(vsi_cmodel/vpusmm)

if (ENABLE_VPUAL)
    if (NOT vpualHost_FOUND)
        if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/movidius/vpualHost")
            message(FATAL_ERROR "vpualHost is not found. Please use: git submodule update --init --recursive to get it")
        endif()

        # including vpualHost/host directory directly not to add vpualHost's examples to the build
        add_subdirectory(movidius/vpualHost/host)

        set(VPUAL_LIBS XLink VpualDispatcher sipp_custom RemoteFlic NN)

        # TODO: remove some day when all warnings are fixed
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set_target_properties(${VPUAL_LIBS} PROPERTIES COMPILE_OPTIONS "-Wno-error"
                 INTERFACE_COMPILE_OPTIONS "-Wno-error=non-virtual-dtor")
        endif()

        # TODO: remove some day when vpualHost will add exports
        set_target_properties(${VPUAL_LIBS} PROPERTIES CXX_VISIBILITY_PRESET default
                                                         C_VISIBILITY_PRESET default)
    endif()

    # copy memory manager library to target directory when vpualHost is enabled
    get_target_property(VPUSMM_LIB_PATH vpusmm IMPORTED_LOCATION)
    set(VPUSMM_OUT_DIR "$<TARGET_FILE_DIR:kmbPlugin>")
    # FIXME: dirty hack to copy vpusmm and symlinks on it
    add_custom_target(vpusmm_copy_library ALL
        COMMAND "${CMAKE_COMMAND}" -E copy "${VPUSMM_LIB_PATH}*" "${VPUSMM_OUT_DIR}"
        COMMENT "[VPUSMM] Copy vpusmm library to build results")

    add_subdirectory(gapi_sipp)

    if (ENABLE_TESTS)
        add_subdirectory(tests/integration)
    endif()
endif()

if (ENABLE_MCM_COMPILER)
    message(STATUS "************ Enabling MCM compiler ************")

    set(MCM_BASE_DIR ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/thirdparty/movidius/mcmCompiler)

    if (NOT ENABLE_PLUGIN_RPATH)
        set(MCM_COMPILER_CMAKE_ARGS -DCMAKE_SKIP_RPATH=ON)
    endif()

    if(DEFINED METIS_DIR)
        set(METIS_INCLUDE_PATH "-DMETIS_INCLUDE_DIR=${METIS_DIR}/include"
                               "-DMETIS_LIBRARY=${METIS_DIR}/lib")
        message(STATUS "METIS_DIR=${METIS_DIR}")
    endif()
    ExternalProject_Add(
            mcmCompiler
            PREFIX ${MCM_BASE_DIR}
            SOURCE_DIR ${MCM_BASE_DIR}
            BINARY_DIR ${MCM_BASE_DIR}/build
            STAMP_DIR ${MCM_BASE_DIR}/stamp
            INSTALL_COMMAND ""
            CMAKE_ARGS ${MCM_COMPILER_CMAKE_ARGS} ${METIS_INCLUDE_PATH})

    set(MCM_COMPILER_INCLUDE_PATH "${MCM_BASE_DIR}"
                                  "${MCM_BASE_DIR}/include"
                                  "${MCM_BASE_DIR}/build/meta" PARENT_SCOPE)
    set(MCM_COMPILER_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cm${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
    set(MCM_COMPILER_MODEL_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}model${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
    set(MCM_COMPILER_METAMODEL_LIBRARY "${MCM_BASE_DIR}/build/lib/${CMAKE_SHARED_LIBRARY_PREFIX}metamodel${CMAKE_SHARED_LIBRARY_SUFFIX}" PARENT_SCOPE)
    set(FLATBUFFERRS_INCLUDE_PATH "${MCM_BASE_DIR}/schema/graphfile/flatbuffers/include/" PARENT_SCOPE)
    set(MCM_SCHEMA_INCLUDE_PATH "${MCM_BASE_DIR}/build/meta/schema/graphfile" PARENT_SCOPE)
else()
    message(STATUS "************ Disabling MCM compiler ************")
endif()
