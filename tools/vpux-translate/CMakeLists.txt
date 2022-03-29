#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME "vpux-translate")

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADD_CLANG_FORMAT
    ENABLE_WARNINGS_AS_ERRORS
    LINK_LIBRARIES
         vpux_utils
         vpux_al
         vpux_elf
         vpux_mlir_compiler_static
         vpux_translate_utils_static
)
