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

set(LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
set(LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm")

set(MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir")
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir")

set(LLVM_INCLUDE_DIRS
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
)

set(MLIR_INCLUDE_DIRS
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

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
    target_compile_options(flatbuffers PRIVATE -Wno-error=suggest-override)
    target_compile_options(flatc PRIVATE -Wno-error=suggest-override)
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
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/src/bitCompactor.cpp")

    target_include_directories(${BITCOMPACTOR_TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/bitcompactor/include>
    )
    if (NOT BUILD_SHARED_LIBS)
        install(TARGETS ${BITCOMPACTOR_TARGET_NAME} EXPORT OpenVINOTargets
                RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT}
                LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${VPUX_PLUGIN_COMPONENT})
    endif()
endif()

#
# huffman_codec
#
if(ENABLE_HUFFMAN_CODEC)
    set(HUFFMAN_CODEC_TARGET_NAME "huffman_codec")
    add_library(${HUFFMAN_CODEC_TARGET_NAME}
        OBJECT
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/compression_codec/huffman_encoding/src/logging.cpp"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/compression_codec/huffman_encoding/src/ms_vasprintf.cpp"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/compression_codec/huffman_encoding/src/Huffman.cpp"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/compression_codec/huffman_encoding/src/huffmanCodec.cpp")

    target_include_directories(${HUFFMAN_CODEC_TARGET_NAME}
        PUBLIC
            "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/compression_codec/huffman_encoding/include"
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
