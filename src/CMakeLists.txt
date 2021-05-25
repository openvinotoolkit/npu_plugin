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

add_subdirectory(utils)
add_subdirectory(vpux_al)
add_subdirectory(vpual_backend)
add_subdirectory(mcmCompiler)
add_subdirectory(frontend_mcm)
add_subdirectory(custom_extension_library)

# TODO Should be only on AARCH64. Enabled for kmbPlugin Windows build
#if(AARCH64)
    add_subdirectory(vpux_plugin/preproc_gapi)
#endif()

if(ENABLE_EMULATOR)
    add_subdirectory(emulator_backend)
endif()
if(ENABLE_HDDL2 AND UNIX)
    add_subdirectory(hddl2_backend)
endif()
if(ENABLE_ZEROAPI_BACKEND)
    add_subdirectory(zero_backend)
endif()
add_subdirectory(vpux_plugin)

add_subdirectory(vpux_utils)
add_subdirectory(vpux_compiler)
