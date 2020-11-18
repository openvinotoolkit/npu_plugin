#
# Copyright 2020 Intel Corporation.
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

if(DEFINED OECORE_BASE_DIR)
    # OECORE_BASE_DIR was passed via CMake command line, nothing to do
elseif(DEFINED ENV{OECORE_BASE_DIR})
    # User sets OECORE_BASE_DIR environment variable
    set(OECORE_BASE_DIR $ENV{OECORE_BASE_DIR})
elseif(DEFINED ENV{OECORE_NATIVE_SYSROOT})
    # OECORE_NATIVE_SYSROOT is a default environment variable for the ORCore toolchain
    set(OECORE_BASE_DIR "$ENV{OECORE_NATIVE_SYSROOT}/../..")
else()
    # Use default value
    set(OECORE_BASE_DIR "/usr/local/oecore-x86_64")
endif()

set(OECORE_TARGET_NAME              "aarch64-ese-linux")
set(OECORE_TARGET_SYSROOT           "${OECORE_BASE_DIR}/sysroots/${OECORE_TARGET_NAME}")
set(OECORE_HOST_SYSROOT             "${OECORE_BASE_DIR}/sysroots/x86_64-esesdk-linux")
set(OECORE_HOST_COMPILER_BIN_DIR    "${OECORE_HOST_SYSROOT}/usr/bin/${OECORE_TARGET_NAME}")

set(CMAKE_SYSTEM_NAME       "Linux")
set(CMAKE_SYSTEM_PROCESSOR  "aarch64")

set(CMAKE_SYSROOT "${OECORE_TARGET_SYSROOT}")

set(CMAKE_C_COMPILER    "${OECORE_HOST_COMPILER_BIN_DIR}/aarch64-ese-linux-gcc")
set(CMAKE_CXX_COMPILER  "${OECORE_HOST_COMPILER_BIN_DIR}/aarch64-ese-linux-g++")

set(CMAKE_C_FLAGS_INIT      "-mcpu=cortex-a53 -mtune=cortex-a53 --sysroot=${OECORE_TARGET_SYSROOT}")
set(CMAKE_CXX_FLAGS_INIT    "-mcpu=cortex-a53 -mtune=cortex-a53 --sysroot=${OECORE_TARGET_SYSROOT}")

set(CMAKE_EXE_LINKER_FLAGS_INIT     "-Wl,-O1 -Wl,--hash-style=gnu -Wl,--as-needed --sysroot=${OECORE_TARGET_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT  "-Wl,-O1 -Wl,--hash-style=gnu -Wl,--as-needed --sysroot=${OECORE_TARGET_SYSROOT}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT  "-Wl,-O1 -Wl,--hash-style=gnu -Wl,--as-needed --sysroot=${OECORE_TARGET_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
