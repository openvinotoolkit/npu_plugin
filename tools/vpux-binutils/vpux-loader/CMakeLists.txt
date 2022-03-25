#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME "vpux-loader")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_executable(${TARGET_NAME} ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tools")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
vpux_enable_clang_format(${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        vpux_elf
        vpux_utils
        vpux_mlir_compiler_static
        vpux_translate_utils_static
)
