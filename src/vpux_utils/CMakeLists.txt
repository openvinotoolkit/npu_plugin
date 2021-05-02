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

set(TARGET_NAME "vpux_utils")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

ie_faster_build(${TARGET_NAME}
    PCH PUBLIC "src/precomp.hpp"
)

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
vpux_enable_clang_format(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${LLVM_INCLUDE_DIRS}
)
target_include_directories(${TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine_plugin_api
    PUBLIC
        IE::inference_engine
        IE::ngraph
        LLVMSupport
)
