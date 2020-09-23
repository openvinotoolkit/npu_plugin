# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "common_tests_helpers")

add_library(${TARGET_NAME} INTERFACE)

target_include_directories(${TARGET_NAME}
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

link_system_libraries(${TARGET_NAME}
    INTERFACE
        IE::inference_engine
        IE::format_reader
)
if(ENABLE_HDDL2_TESTS)
    link_system_libraries(${TARGET_NAME}
        INTERFACE
            HddlUnite
    )
endif()
