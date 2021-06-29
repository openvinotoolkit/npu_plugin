# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

include(ExternalProject)

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

if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

#
# OpenCL compiler
#

if(LINUX OR WIN32)
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

        if(LINUX)
            set(VPU_CLC_MA2X9X_VERSION "movi-cltools-21.03.1-linux64")
            RESOLVE_DEPENDENCY(VPU_CLC_MA2X9X
                ARCHIVE_LIN "VPU_OCL_compiler/${VPU_CLC_MA2X9X_VERSION}.tar.gz"
                TARGET_PATH "${TEMP}/vpu/clc/ma2x9x/${VPU_CLC_MA2X9X_VERSION}"
                ENVIRONMENT "VPU_CLC_MA2X9X_ROOT"
                SHA256 "db8ef81ab172a8b627a667109a1c89ec6ab0f8ddb4a5bcead1fdbbbdc5d6a11a")
        elseif(WIN32)
            set(VPU_CLC_MA2X9X_VERSION "movi-cltools-21.03.1-win64")
            set(HOST_EXECUTABLE_SUFFIX ".exe")
            RESOLVE_DEPENDENCY(VPU_CLC_MA2X9X
                ARCHIVE_WIN "VPU_OCL_compiler/${VPU_CLC_MA2X9X_VERSION}.tar.gz"
                TARGET_PATH "${TEMP}/vpu/clc/ma2x9x/${VPU_CLC_MA2X9X_VERSION}"
                ENVIRONMENT "VPU_CLC_MA2X9X_ROOT"
                SHA256 "d6343ec74d2d43c1725023061adbcba9431f1966a76456bfd8351da46a601a32")
        endif()
        debug_message(STATUS "VPU_CLC_MA2X9X=" ${VPU_CLC_MA2X9X})

        update_deps_cache(
            VPU_CLC_MA2X9X_ROOT
            "${VPU_CLC_MA2X9X}"
            "[KMB] Root directory of OpenCL compiler")

        update_deps_cache(
            VPU_CLC_MA2X9X_COMMAND
            "${VPU_CLC_MA2X9X}/bin/clc${HOST_EXECUTABLE_SUFFIX}"
            "[KMB] OpenCL compiler")

        find_program(VPU_CLC_MA2X9X_COMMAND clc)

        unset(IE_PATH_TO_DEPS)
    endif()
endif()

#
# `kmb_custom_ocl_kernels` CMake target
#

add_library(kmb_custom_ocl_kernels INTERFACE)

function(add_kmb_compile_custom_ocl_kernels KMB_SRC_DIR)
    set(SRC_DIR "${CMAKE_SOURCE_DIR}/src/${KMB_SRC_DIR}")
    set(DST_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/kmb_${KMB_SRC_DIR}")

    add_custom_target(make_${KMB_SRC_DIR}_dst_dir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DST_DIR}"
        VERBATIM
     )

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
            DEPENDS make_${KMB_SRC_DIR}_dst_dir
            COMMENT "[KMB] Copy ${xml_file} to ${DST_DIR}"
            VERBATIM)
    endforeach()

    foreach(cl_file IN LISTS CL_FILES)
        get_filename_component(cl_file_name ${cl_file} NAME_WE)

        set(out_file "${DST_DIR}/${cl_file_name}.bin")
        list(APPEND all_output_files ${out_file})

        if(LINUX)
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
                DEPENDS make_${KMB_SRC_DIR}_dst_dir ${VPU_CLC_MA2X9X_COMMAND}
                COMMENT "[KMB] Compile ${cl_file}"
                VERBATIM)
        elseif(WIN32)
            add_custom_command(
                OUTPUT ${out_file}
                COMMAND
                    ${CMAKE_COMMAND} -E env
                        "SHAVE_LDSCRIPT_DIR=${VPU_CLC_MA2X9X}/ldscripts/3010xx"
                        "SHAVE_MA2X8XLIBS_DIR=${VPU_CLC_MA2X9X}/lib"
                        "SHAVE_LLD_DIR=${VPU_CLC_MA2X9X}/bin"
                    ${VPU_CLC_MA2X9X_COMMAND} --strip-binary-header -d 3010xx --use-lld ${cl_file} -o ${out_file}
                MAIN_DEPENDENCY ${cl_file}
                DEPENDS make_${KMB_SRC_DIR}_dst_dir ${VPU_CLC_MA2X9X_COMMAND}
                COMMENT "[KMB] Compile ${cl_file}"
                VERBATIM)
        endif()
    endforeach()

    add_custom_target("kmb_compile_${KMB_SRC_DIR}"
        DEPENDS ${all_output_files}
        COMMENT "[KMB] Compile ${SRC_DIR}")

    add_dependencies(kmb_custom_ocl_kernels "kmb_compile_${KMB_SRC_DIR}")
endfunction()

if(VPU_CLC_MA2X9X_COMMAND)
    add_kmb_compile_custom_ocl_kernels("custom_ocl_kernels")
endif()

if(VPU_CLC_MA2X9X_COMMAND OR CMAKE_CROSSCOMPILING)
    target_compile_definitions(kmb_custom_ocl_kernels INTERFACE "KMB_HAS_CUSTOM_OCL_KERNELS")
endif()

#
# `kmb_custom_cpp_kernels` CMake target
#

add_library(kmb_custom_cpp_kernels INTERFACE)

function(add_kmb_compile_custom_cpp_kernels)
    set(BUILD_COMMAND "${CMAKE_SOURCE_DIR}/src/custom_cpp_kernels/tools/build_kernel.py")

    set(SRC_DIR "${CMAKE_SOURCE_DIR}/src/custom_cpp_kernels")
    set(DST_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/kmb_custom_cpp_kernels")

    add_custom_target(make_custom_cpp_kernels_dst_dir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DST_DIR}"
        VERBATIM
    )

    file(GLOB XML_FILES "${SRC_DIR}/*.xml")
    file(GLOB CPP_FILES "${SRC_DIR}/*.cpp")
    file(GLOB ELF_FILES "${SRC_DIR}/*.elf")

    foreach(xml_file IN LISTS XML_FILES)
        get_filename_component(xml_file_name ${xml_file} NAME)

        set(out_file "${DST_DIR}/${xml_file_name}")
        list(APPEND all_output_files ${out_file})

        add_custom_command(
                OUTPUT ${out_file}
                COMMAND
                ${CMAKE_COMMAND} -E copy ${xml_file} ${out_file}
                MAIN_DEPENDENCY ${xml_file}
                DEPENDS make_custom_cpp_kernels_dst_dir
                COMMENT "[KMB] Copy ${xml_file} to ${DST_DIR}"
                VERBATIM)
    endforeach()

    if (DEFINED MV_TOOLS_PATH)
        foreach(cpp_file IN LISTS CPP_FILES)
            get_filename_component(cpp_file_name ${cpp_file} NAME_WE)

            set(out_file "${DST_DIR}/${cpp_file_name}.elf")
            list(APPEND all_output_files ${out_file})

            add_custom_command(
                    OUTPUT ${out_file}
                    COMMAND
                    python3 ${BUILD_COMMAND} --i ${cpp_file}  --t "${MV_TOOLS_PATH}" --o ${out_file}
                    MAIN_DEPENDENCY ${elf_file}
                    DEPENDS make_custom_cpp_kernels_dst_dir
                    COMMENT "[KMB] Compile ${cpp_file}"
                    VERBATIM)
        endforeach()
    else()
        foreach(elf_file IN LISTS ELF_FILES)
            get_filename_component(elf_file_name ${elf_file} NAME)

            set(out_file "${DST_DIR}/${elf_file_name}")
            list(APPEND all_output_files ${out_file})

            add_custom_command(
                    OUTPUT ${out_file}
                    COMMAND
                    ${CMAKE_COMMAND} -E copy ${elf_file} ${out_file}
                    MAIN_DEPENDENCY ${elf_file}
                    DEPENDS make_custom_cpp_kernels_dst_dir
                    COMMENT "[KMB] Copy ${elf_file} to ${DST_DIR}"
                    VERBATIM)
        endforeach()
    endif()

    add_custom_target(kmb_compile_custom_cpp_kernels
            DEPENDS ${all_output_files}
            COMMENT "[KMB] Compile custom C++ kernels")

    add_dependencies(kmb_custom_cpp_kernels kmb_compile_custom_cpp_kernels)
    target_compile_definitions(kmb_custom_cpp_kernels INTERFACE "KMB_HAS_CUSTOM_CPP_KERNELS")
endfunction()

if (NOT DEFINED MV_TOOLS_PATH)
    if(DEFINED MV_TOOLS_DIR AND DEFINED MV_TOOLS_VERSION)
        set(MV_TOOLS_PATH ${MV_TOOLS_DIR}/${MV_TOOLS_VERSION})
    endif()
endif()

add_kmb_compile_custom_cpp_kernels()

#
# `kmb_custom_extension` CMake target
#

add_library(kmb_custom_extension INTERFACE)

if(VPU_CLC_MA2X9X_COMMAND)
    add_kmb_compile_custom_ocl_kernels("custom_extension_library")
endif()

#
# HDDLUnite
#

if(ENABLE_HDDL2 AND UNIX)
    set(PCIE_DRIVERS_KMB_ARCHIVE_VERSION RELEASE_ww17_2021)
    set(PCIE_DRIVERS_KMB_ARCHIVE_HASH "3d155615b6768d5b5f55a243cde4886408862180a5908c0805e699d59e317bf8")

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(FATAL_ERROR "HDDLUnite is not found (missing THIRDPARTY_SERVER_PATH).")
    endif()

    if(DEFINED IE_PATH_TO_DEPS)
        reset_deps_cache(PCIE_DRIVERS)

        RESOLVE_DEPENDENCY(PCIE_DRIVERS
                ARCHIVE_LIN "hddl2/kmb-pcie-drivers_${PCIE_DRIVERS_KMB_ARCHIVE_VERSION}.tgz"
                ENVIRONMENT "PCIE_DRIVERS"
                TARGET_PATH "${TEMP}/pcie_drivers"
                SHA256 ${PCIE_DRIVERS_KMB_ARCHIVE_HASH})

        unset(IE_PATH_TO_DEPS)
    endif()

    if(NOT ENABLE_CUSTOM_HDDLUNITE)
        set(HDDLUNITE_KMB_ARCHIVE_VERSION RELEASE_ww17.2_2021)
        set(HDDLUNITE_KMB_ARCHIVE_HASH "d589b327e1a4c13a3a0e869730415a4456d058aab8b6486e66fb0d2c6706939c")
        set(HDDLUNITE_VPUX_4_ARCHIVE_VERSION RELEASE_VPUX_4_ww24_2021)
        set(HDDLUNITE_VPUX_4_ARCHIVE_HASH "a6c89dffa4e1c3c2fef652f09c536d17c5df6532c68c7fb7c842098f0b307b6f")
        set(ARCH_FORMAT ".tgz")

        if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
            set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        elseif(DEFINED THIRDPARTY_SERVER_PATH)
            set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        else()
            message(FATAL_ERROR "HDDLUnite is not found (missing THIRDPARTY_SERVER_PATH).")
        endif()

        if(DEFINED IE_PATH_TO_DEPS)
            reset_deps_cache(HDDL_UNITE)
            reset_deps_cache(HDDL_UNITE_VPUX_4)

            RESOLVE_DEPENDENCY(HDDL_UNITE
                    ARCHIVE_LIN "hddl_unite/hddl_unite_${HDDLUNITE_KMB_ARCHIVE_VERSION}${ARCH_FORMAT}"
                    ENVIRONMENT "HDDL_UNITE"
                    TARGET_PATH "${TEMP}/hddl_unite"
                    SHA256 ${HDDLUNITE_KMB_ARCHIVE_HASH})

            RESOLVE_DEPENDENCY(HDDL_UNITE_VPUX_4
                    ARCHIVE_LIN "hddl_unite/hddl_unite_${HDDLUNITE_VPUX_4_ARCHIVE_VERSION}${ARCH_FORMAT}"
                    ENVIRONMENT "HDDL_UNITE_VPUX_4"
                    TARGET_PATH "${TEMP}/vpux_4/hddl_unite"
                    SHA256 ${HDDLUNITE_VPUX_4_ARCHIVE_HASH})

            unset(IE_PATH_TO_DEPS)
        endif()

        find_library(HDDL_UNITE_LIBRARY
            NAMES HddlUnite
            HINTS "${HDDL_UNITE}/lib"
            NO_DEFAULT_PATH)

        log_rpath(HDDL_UNITE "${HDDL_UNITE_LIBRARY}")

        add_library(HddlUnite SHARED IMPORTED GLOBAL)

        set_target_properties(HddlUnite PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HDDL_UNITE}/include"
            IMPORTED_LOCATION ${HDDL_UNITE_LIBRARY}
            IMPORTED_NO_SONAME TRUE)

        find_library(XLINK_LIBRARY
            NAMES XLink
            HINTS "${HDDL_UNITE}/thirdparty/XLink/lib"
            NO_DEFAULT_PATH)

        add_library(HDDLUniteXLink SHARED IMPORTED GLOBAL)

        set_target_properties(HDDLUniteXLink PROPERTIES
            IMPORTED_LOCATION ${XLINK_LIBRARY}
            IMPORTED_NO_SONAME TRUE)

        set(XLINK_LIB HDDLUniteXLink CACHE INTERNAL "")

    endif()
endif()
