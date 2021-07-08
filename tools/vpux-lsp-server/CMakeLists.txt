#
# Copyright Intel Corporation.
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

set(TARGET_NAME "vpux-lsp-server")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_executable(${TARGET_NAME} ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tools")

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
vpux_enable_clang_format(${TARGET_NAME})

link_system_libraries(${TARGET_NAME}
    PRIVATE
        MLIRLspServerLib
)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        vpux_compiler_static
)
