#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME "vpux-loader")

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ENABLE_WARNINGS_AS_ERRORS
    ADD_CLANG_FORMAT
    LINK_LIBRARIES
         vpux_elf
         vpux_utils
         vpux_mlir_compiler_static
         vpux_translate_utils_static
)
