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

set(TARGET_NAME elfUnitTests)

file(GLOB_RECURSE UNIT_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${UNIT_SOURCES})

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        elf
        IE::commonTestUtils
    LABELS
        elf
)
