#
# Copyright 2020 Intel Corporation.
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

replace_noerror()

set(TARGET_NAME_RAW "vpux_utils_raw")
set(TARGET_NAME "vpux_utils")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME_RAW} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME_RAW} PROPERTIES FOLDER "src")

ie_faster_build(${TARGET_NAME_RAW}
    PCH PUBLIC "src/precomp.hpp"
)

enable_warnings_as_errors(${TARGET_NAME_RAW} WIN_STRICT)
vpux_enable_clang_format(${TARGET_NAME_RAW})

target_include_directories(${TARGET_NAME_RAW}
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${LLVM_INCLUDE_DIRS}
)

target_include_directories(${TARGET_NAME_RAW}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/include/vpux/compiler/dialect/VPUIP/generated>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# FIXME: S#71607
if(NOT MSVC)
    target_compile_options(${TARGET_NAME_RAW}
        PUBLIC
            -Wno-error=deprecated-declarations
    )
endif()

link_system_libraries(${TARGET_NAME_RAW}
    PRIVATE
        IE::inference_engine_plugin_api
    PUBLIC
        IE::inference_engine
        IE::ngraph
        LLVMSupport
        flatbuffers
)

add_dependencies(${TARGET_NAME_RAW}
    vpux_compiler_schema
)

# make vpux_utils include LLVMSupport and depending on it libraries
# TODO: this target needs to be installed
bundle_static_library(${TARGET_NAME_RAW} ${TARGET_NAME} LLVMSupport)

find_package(ZLIB)
target_link_libraries(${TARGET_NAME}
    INTERFACE
        IE::inference_engine
        $<TARGET_PROPERTY:LLVMSupport,INTERFACE_LINK_LIBRARIES>
)

