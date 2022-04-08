#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME "vpuxUnitTests")
set(EXCLUDED_UNIT_TESTS_DIR "")
set(OPTIONAL_UNIT_TESTS_INCLUDES "")
set(OPTIONAL_UNIT_TESTS_LIBS "")

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    list(APPEND OPTIONAL_UNIT_TESTS_INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_driver_compiler_adapter"
    )

    list(APPEND OPTIONAL_UNIT_TESTS_LIBS
            vpux_driver_compiler_adapter_static
        )
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_driver_compiler_adapter"
    )
endif()

#
# Add optional tests on HuffmanCodec
#
if(ENABLE_HUFFMAN_CODEC)
    list(APPEND OPTIONAL_UNIT_TESTS_INCLUDES
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/src/compression_codec/huffman_encoding/include"
    )

    list(APPEND OPTIONAL_UNIT_TESTS_LIBS
        huffman_codec
    )
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/huffman_codec"
    )
endif()

add_subdirectory(kmb/test_utils)

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_UNIT_TESTS_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/kmb/test_utils
    INCLUDES
        ${OPTIONAL_UNIT_TESTS_INCLUDES}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include"
        ${VPUAL_INCLUDES}
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/artifacts/vpuip_2"
    LINK_LIBRARIES
        ${OPTIONAL_UNIT_TESTS_LIBS}
        vpux_elf
        IE::commonTestUtils
        IE::ngraph
        IE::ngraphFunctions
        IE::gmock
        common_tests_helpers
        ${VPUX_PLUGIN_STATIC_TARGET}
        kmb_utils
        vpux_al
        vpux_utils
        vpux_mlir_compiler_static
        vpux_translate_utils_static
    LINK_LIBRARIES_WHOLE_ARCHIVE
        kmbTestData
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tests")
add_dependencies(${TARGET_NAME} throw_test_backend vpu3700_test_backend no_devices_test_backend)

enable_warnings_as_errors(${TARGET_NAME})
vpux_enable_clang_format(${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

