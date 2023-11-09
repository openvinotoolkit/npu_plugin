#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
# Replace compiler flags
#

foreach(flag IN ITEMS "-Werror" "-fvisibility=hidden" "-fvisibility-inlines-hidden")
    string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE ${flag} "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE ${flag} "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endforeach()


# TODO: Remove once the thirdparties respect these flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(ie_c_cxx_deprecated_no_errors "${ie_c_cxx_deprecated_no_errors} /wd4146 /wd4703")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
endif()

set(BUILD_SHARED_LIBS OFF)

#
# LLVM/MLIR
#

set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_EH ON CACHE BOOL "" FORCE)
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR ENABLE_DEVELOPER_BUILD)
    set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
else()
    set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "" FORCE)
endif()
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(CROSS_TOOLCHAIN_FLAGS_ "" CACHE STRING "" FORCE)
set(CROSS_TOOLCHAIN_FLAGS_NATIVE "" CACHE STRING "" FORCE)
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)
# we do not use examples and having it enabled
# makes cmake complains about long path on Windows
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(llvm-project/llvm EXCLUDE_FROM_ALL)

set(MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir/include")
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir/include")
include_directories(SYSTEM
    ${MLIR_SOURCE_DIR}
    ${MLIR_BINARY_DIR})

#
# flatbuffers
#

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)

add_subdirectory(flatbuffers EXCLUDE_FROM_ALL)

# FIXME: this flag is forced while building in a single tree with OV
# but code cannot be compiled with the flag
if(NOT MSVC)
    target_compile_options(flatbuffers PRIVATE -Wno-suggest-override)
    target_compile_options(flatc PRIVATE -Wno-suggest-override)
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13))
        target_compile_options(flatbuffers PRIVATE -Wno-unused-but-set-variable)
        target_compile_options(flatc PRIVATE -Wno-unused-but-set-variable)
    endif()
endif()

vpux_add_native_tool(flatc "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers"
    CMAKE_ARGS
        "FLATBUFFERS_BUILD_TESTS:BOOL=OFF"
        "FLATBUFFERS_INSTALL:BOOL=OFF"
        "FLATBUFFERS_BUILD_FLATC:BOOL=ON"
)

#
# vpux_elf
#

set(VPUX_ELF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/elf")
set(VPUX_ELF_SOURCE_DIR "${VPUX_ELF_SOURCE_DIR}" PARENT_SCOPE)
message("VPUX_ELF_SOURCE_DIR : ${VPUX_ELF_SOURCE_DIR}")

add_subdirectory(${VPUX_ELF_SOURCE_DIR}/vpux_elf)
target_include_directories(vpux_elf PRIVATE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/dialect/VPU37XX")

#
# zeroApi
#

if(ENABLE_ZEROAPI_BACKEND)
    # We have to avoid linking against loader with debug postfix due it's a part of driver
    # and the name will be the same for release and debug configurations
    set(OLD_CMAKE_DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set(CMAKE_DEBUG_POSTFIX "")
    add_subdirectory(level-zero EXCLUDE_FROM_ALL)
    set(CMAKE_DEBUG_POSTFIX ${OLD_CMAKE_DEBUG_POSTFIX})
endif()

#
# bitcompactor
#

if(ENABLE_BITCOMPACTOR)
    set(BITCOMPACTOR_TARGET_NAME "bitcompactor")

    add_library(${BITCOMPACTOR_TARGET_NAME}
        OBJECT
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/src/bitCompactor.cpp"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/src/utils/logger.cpp"
        )

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_include_directories(${BITCOMPACTOR_TARGET_NAME}
        PUBLIC
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/include"
    )
endif()

#
# nn cost model
#

set(VPUNN_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(VPUNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VPUNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(vpucostmodel)

add_subdirectory(yaml-cpp)
# NB: Suppress warnings in yaml-cpp
if(NOT MSVC)
    target_compile_options(yaml-cpp PRIVATE -Wno-suggest-override)
endif()
