# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(models)
include(ExternalProject)

set_temp_directory(TEMP "${CMAKE_SOURCE_DIR}")

set(MODELS_PATH "${TEMP}/models")
debug_message(STATUS "MODELS_PATH=" ${MODELS_PATH})

set(DATA_PATH "${TEMP}/validation_set/src/validation_set")
debug_message(STATUS "DATA_PATH=" ${DATA_PATH})

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

include(dependency_solver)

include(linux_name)
if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

#
# OpenCL compiler
#
set(VPU_CLC_MA2X9X_VERSION "movi-cltools-20.03.18")

if(LINUX AND LINUX_OS_NAME MATCHES "Ubuntu")
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(WARNING "VPU_OCL_COMPILER is not found. Some tests will skipped")
    endif()

    if(DEFINED IE_PATH_TO_DEPS)
        message(STATUS "THIRDPARTY_SERVER_PATH=${IE_PATH_TO_DEPS}")

        reset_deps_cache(VPU_CLC_MA2X9X_ROOT)
        reset_deps_cache(VPU_CLC_MA2X9X_COMMAND)

        RESOLVE_DEPENDENCY(VPU_CLC_MA2X9X
            ARCHIVE_LIN "VPU_OCL_compiler/${VPU_CLC_MA2X9X_VERSION}.tar.gz"
            TARGET_PATH "${TEMP}/vpu/clc/ma2x9x/${VPU_CLC_MA2X9X_VERSION}"
            ENVIRONMENT "VPU_CLC_MA2X9X_COMMAND")
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
#        unset(IE_PATH_TO_DEPS)
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
                    "SHAVE_LDSCRIPT_DIR=${VPU_CLC_MA2X9X}/ldscripts/"
                    "SHAVE_MA2X8XLIBS_DIR=${VPU_CLC_MA2X9X}/lib"
                    "SHAVE_MOVIASM_DIR=${VPU_CLC_MA2X9X}/bin"
                    "SHAVE_MYRIAD_LD_DIR=${VPU_CLC_MA2X9X}/bin"
                ${VPU_CLC_MA2X9X_COMMAND} --strip-binary-header ${cl_file} -o ${out_file}
            MAIN_DEPENDENCY ${cl_file}
            DEPENDS ${VPU_CLC_MA2X9X_COMMAND}
            COMMENT "[KMB] Compile ${cl_file}"
            VERBATIM)
    endforeach()

    add_custom_target(kmb_compile_custom_kernels
        DEPENDS ${all_output_files}
        COMMENT "[KMB] Compile custom kernels")

    add_dependencies(kmb_custom_kernels kmb_compile_custom_kernels)
    target_compile_definitions(kmb_custom_kernels INTERFACE "KMB_HAS_CUSTOM_KERNELS")
endfunction()

if(VPU_CLC_MA2X9X_COMMAND)
    add_kmb_compile_custom_kernels()
endif()
