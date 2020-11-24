# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(ExternalProject)

include(models)
include(dependency_solver)
include(linux_name)

if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

set_temp_directory(TEMP "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}")

# FIXME: Create empty file to avoid errors on CI
file(TOUCH "${CMAKE_BINARY_DIR}/ld_library_rpath_64.txt")

#
# Models and Images for tests
#

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=${MODELS_PATH}")

set(DATA_PATH "${TEMP}/validation_set/src/validation_set")
debug_message(STATUS "DATA_PATH=${DATA_PATH}")

add_models_repo(${ENABLE_MODELS} "models:git@gitlab-icv.inn.intel.com:inference-engine/models-ir.git")

if (ENABLE_VALIDATION_SET)
    add_lfs_repo(
        "validation_set"
        "${TEMP}/validation_set"
        "git@gitlab-icv.inn.intel.com:inference-engine/validation-set.git"
        "master"
    )
endif()

fetch_models_and_validation_set()

#
# OpenCL compiler
#

set(VPU_CLC_MA2X9X_VERSION "movi-cltools-20.09.2")

if(LINUX AND LINUX_OS_NAME MATCHES "Ubuntu")
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(WARNING "VPU_OCL_COMPILER is not found (missing THIRDPARTY_SERVER_PATH). Some tests will be skipped.")
    endif()

    if(DEFINED IE_PATH_TO_DEPS)
        debug_message(STATUS "THIRDPARTY_SERVER_PATH=${IE_PATH_TO_DEPS}")

        reset_deps_cache(VPU_CLC_MA2X9X_ROOT)
        reset_deps_cache(VPU_CLC_MA2X9X_COMMAND)

        RESOLVE_DEPENDENCY(VPU_CLC_MA2X9X
            ARCHIVE_LIN "VPU_OCL_compiler/${VPU_CLC_MA2X9X_VERSION}.tar.gz"
            TARGET_PATH "${TEMP}/vpu/clc/ma2x9x/${VPU_CLC_MA2X9X_VERSION}"
            ENVIRONMENT "VPU_CLC_MA2X9X_COMMAND"
            SHA256 "0a864bd0e11cee2d85ac7e451dddae19216c8bc9bb50e1a8e0151ab97d5e3c8d")
        debug_message(STATUS "VPU_CLC_MA2X9X=" ${VPU_CLC_MA2X9X})

        update_deps_cache(
            VPU_CLC_MA2X9X_ROOT
            "${VPU_CLC_MA2X9X}"
            "[KMB] Root directory of OpenCL compiler")

        update_deps_cache(
            VPU_CLC_MA2X9X_COMMAND
            "${VPU_CLC_MA2X9X}/bin/clc"
            "[KMB] OpenCL compiler")

        find_program(VPU_CLC_MA2X9X_COMMAND clc)
        unset(IE_PATH_TO_DEPS)
    endif()
endif()

#
# `kmb_custom_kernels` CMake target
#

add_library(kmb_custom_kernels INTERFACE)

function(add_kmb_compile_custom_kernels)
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/src/custom_kernels")
    set(DST_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/kmb_custom_kernels")

    file(MAKE_DIRECTORY "${DST_DIR}")

    file(GLOB XML_FILES "${SRC_DIR}/*.xml")
    file(GLOB CL_FILES "${SRC_DIR}/*.cl")

    foreach(xml_file IN LISTS XML_FILES)
        get_filename_component(xml_file_name ${xml_file} NAME)

        set(out_file "${DST_DIR}/${xml_file_name}")
        list(APPEND all_output_files ${out_file})

        add_custom_command(
            OUTPUT ${out_file}
            COMMAND
                ${CMAKE_COMMAND} -E copy ${xml_file} ${out_file}
            MAIN_DEPENDENCY ${xml_file}
            COMMENT "[KMB] Copy ${xml_file} to ${DST_DIR}"
            VERBATIM)
    endforeach()

    foreach(cl_file IN LISTS CL_FILES)
        get_filename_component(cl_file_name ${cl_file} NAME_WE)

        set(out_file "${DST_DIR}/${cl_file_name}.bin")
        list(APPEND all_output_files ${out_file})

        add_custom_command(
            OUTPUT ${out_file}
            COMMAND
                ${CMAKE_COMMAND} -E env
                    "SHAVE_LDSCRIPT_DIR=${VPU_CLC_MA2X9X}/ldscripts/3010xx"
                    "SHAVE_MA2X8XLIBS_DIR=${VPU_CLC_MA2X9X}/lib"
                    "SHAVE_MYRIAD_LD_DIR=${VPU_CLC_MA2X9X}/bin"
                    "SHAVE_MOVIASM_DIR=${VPU_CLC_MA2X9X}/bin"
                ${VPU_CLC_MA2X9X_COMMAND} --strip-binary-header -d 3010xx ${cl_file} -o ${out_file}
            MAIN_DEPENDENCY ${cl_file}
            DEPENDS ${VPU_CLC_MA2X9X_COMMAND}
            COMMENT "[KMB] Compile ${cl_file}"
            VERBATIM)
    endforeach()

    add_custom_target(kmb_compile_custom_kernels
        DEPENDS ${all_output_files}
        COMMENT "[KMB] Compile custom kernels")

    add_dependencies(kmb_custom_kernels kmb_compile_custom_kernels)
endfunction()

if(VPU_CLC_MA2X9X_COMMAND)
    add_kmb_compile_custom_kernels()
endif()

if(VPU_CLC_MA2X9X_COMMAND OR CMAKE_CROSSCOMPILING)
    target_compile_definitions(kmb_custom_kernels INTERFACE "KMB_HAS_CUSTOM_KERNELS")
endif()

#
# HDDLUnite
#

if(ENABLE_HDDL2)
    if(UNIX)
        set(HDDLUNITE_KMB_ARCHIVE_VERSION RELEASE_ww45)
        set(HDDLUNITE_TBH_ARCHIVE_VERSION RELEASE_TBH_ww44)
        set(HDDLUNITE_KMB_ARCHIVE_HASH "04df19c47aa4b11e3254f997e892c00eb29ac2e1ce61846282eb462401412077")
        set(HDDLUNITE_TBH_ARCHIVE_HASH "cebc6864a6d20c435d379f6757e9ffea38eea17073bb5363db85a098b476a84d")
        set(ARCH_FORMAT ".tgz")
    else()
        set(HDDLUNITE_KMB_ARCHIVE_VERSION RELEASE_ww44_Windows)
        set(HDDLUNITE_KMB_ARCHIVE_HASH "f1da749d21b6c9190443a3dc681e65397cd1270f44003140dfb8d1350b7ef1e0")
        set(ARCH_FORMAT ".zip")
    endif()

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(FATAL_ERROR "HDDLUnite is not found (missing THIRDPARTY_SERVER_PATH).")
    endif()

    if(DEFINED IE_PATH_TO_DEPS)
        reset_deps_cache(HDDL_UNITE)

        RESOLVE_DEPENDENCY(HDDL_UNITE
                ARCHIVE_LIN "hddl_unite/hddl_unite_${HDDLUNITE_KMB_ARCHIVE_VERSION}${ARCH_FORMAT}"
                ENVIRONMENT "HDDL_UNITE"
                TARGET_PATH "${TEMP}/hddl_unite"
                SHA256 ${HDDLUNITE_KMB_ARCHIVE_HASH})
        if(UNIX)
            RESOLVE_DEPENDENCY(HDDL_UNITE_TBH
                    ARCHIVE_LIN "hddl_unite/hddl_unite_${HDDLUNITE_TBH_ARCHIVE_VERSION}${ARCH_FORMAT}"
                    ENVIRONMENT "HDDL_UNITE_TBH"
                    TARGET_PATH "${TEMP}/tbh/hddl_unite"
                    SHA256 ${HDDLUNITE_TBH_ARCHIVE_HASH})
        endif()

        unset(IE_PATH_TO_DEPS)
    endif()

    find_library(HDDL_UNITE_LIBRARY
        NAMES HddlUnite
        HINTS "${HDDL_UNITE}/lib"
        NO_DEFAULT_PATH)

    log_rpath(HDDL_UNITE "${HDDL_UNITE_LIBRARY}")

    if (WIN32)
        add_library(HddlUnite STATIC IMPORTED GLOBAL)
    else()
        add_library(HddlUnite SHARED IMPORTED GLOBAL)
    endif()

    set_target_properties(HddlUnite PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HDDL_UNITE}/include"
        IMPORTED_LOCATION ${HDDL_UNITE_LIBRARY}
        IMPORTED_NO_SONAME TRUE)

    if(UNIX)
        find_library(XLINK_LIBRARY
            NAMES XLink
            HINTS "${HDDL_UNITE}/thirdparty/XLink/lib"
            NO_DEFAULT_PATH)

        add_library(HDDLUniteXLink SHARED IMPORTED GLOBAL)

        set_target_properties(HDDLUniteXLink PROPERTIES
            IMPORTED_LOCATION ${XLINK_LIBRARY}
            IMPORTED_NO_SONAME TRUE)

        set(XLINK_LIB HDDLUniteXLink CACHE INTERNAL "")
    else()
        set(XLINK_LIB "" CACHE INTERNAL "")
    endif()
endif()
