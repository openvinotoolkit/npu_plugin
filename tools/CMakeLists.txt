#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_subdirectory(model-comparator)

add_subdirectory(test_classification)
add_subdirectory(single-image-test)
add_subdirectory(sol-generator)
add_subdirectory(query_model)

add_subdirectory(vpux-opt)
add_subdirectory(vpux-translate)
add_subdirectory(vpux-lsp-server)

add_subdirectory(profiling_parser)

add_subdirectory(vpux-binutils)

add_subdirectory(legacy/single-image-test)


#
# install python tools
#

set(PYTHON_TOOLS
    "edit-xml"
    "ir-analyzer"
    "model-output-comparator"
)

foreach(TOOL IN LISTS PYTHON_TOOLS)
        install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${TOOL}/"
                DESTINATION "tools/${TOOL}"
                COMPONENT tests
                EXCLUDE_FROM_ALL)

        # TODO: Remove duplication E#31024
        install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${TOOL}/"
                DESTINATION "tools/${TOOL}"
                COMPONENT ${VPUX_TESTS_COMPONENT}
                EXCLUDE_FROM_ALL)
endforeach()
