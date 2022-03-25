#
# Copyright 2019-2021 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

if(BUILD_COMPILER_FOR_DRIVER)
    set(COMPILER_FOR_DRIVER_VERSION_FILE ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_al/src/version.cpp)
endif()

add_subdirectory(utils)
add_subdirectory(vpux_al)

if(AARCH64)
    add_subdirectory(vpual_backend)
endif()

if(ENABLE_MCM_COMPILER)
    add_subdirectory(mcmCompiler)
    add_subdirectory(frontend_mcm)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/custom_extension_library")
    add_subdirectory(custom_extension_library)
endif()

add_subdirectory(vpux_ngraph_transformations)

# TODO Should be only on AARCH64. Enabled for kmbPlugin Windows build
#if(AARCH64)
    add_subdirectory(vpux_plugin/preproc_gapi)
#endif()

if(ENABLE_EMULATOR)
    FetchContent_Declare(
        vpux_emulator
        GIT_REPOSITORY https://github.com/intel-innersource/applications.ai.vpu-accelerators.vpux-emulator.git
        GIT_TAG        afd7e76b9cf9ef1dfd451f3502579c1bdaf1c457
    )
    FetchContent_MakeAvailable(vpux_emulator)

    add_dependencies(vpux_emulator mcmCompilerGenSchema)
    add_subdirectory(emulator_backend)
endif()

if(ENABLE_HDDL2 AND UNIX AND LINUX_OS_NAME STREQUAL "Ubuntu 20.04")
    add_subdirectory(hddl2_backend)
endif()
if(ENABLE_ZEROAPI_BACKEND)
    add_subdirectory(zero_backend)
endif()

if(BUILD_COMPILER_FOR_DRIVER)
    add_subdirectory(VPUXCompilerL0)
endif()
add_subdirectory(vpux_plugin)

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    add_subdirectory(vpux_driver_compiler_adapter)
endif()

add_subdirectory(vpux_utils)

add_subdirectory(vpux_compiler)

if(BUILD_SHARED_LIBS)
    add_subdirectory(vpux_translate_utils)
endif()

if(ENABLE_IMD_BACKEND)
    add_subdirectory(vpux_imd_backend)
endif()
