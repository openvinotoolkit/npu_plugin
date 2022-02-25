#
# Copyright 2021 Intel Corporation.
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

set (EXAMPLES)

foreach(TARGET_NAME ${EXAMPLES})
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TARGET_NAME}.cpp)
    add_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}/${TARGET_NAME}.cpp)

    enable_warnings_as_errors(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} vpux_elf)
endforeach()
