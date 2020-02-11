# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_HDDL2)
    add_subdirectory(hddl2_tests)
endif()

set(TARGET_NAME "KmbFunctionalTests")

file(GLOB_RECURSE KMB_TEST_SOURCES *.cpp *.hpp *.h)
list(FILTER KMB_TEST_SOURCES EXCLUDE REGEX ".*/hddl2_tests/.*")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${KMB_TEST_SOURCES})
endif()

list(APPEND KMB_TEST_LIBRARIES IE::IESharedTests IE::vpu_common_lib IE::ngraph vpusmm kmb_utils)
if(ENABLE_VPUAL AND ENABLE_VPUAL_MODEL)
    list(APPEND KMB_TEST_LIBRARIES vpualModel)
endif()

add_executable(${TARGET_NAME} ${KMB_TEST_SOURCES})

add_dependencies(${TARGET_NAME} kmbPlugin)

target_compile_options(${TARGET_NAME}
    PRIVATE
        -Wall
        -Werror)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        USE_KMB=1
        INSTANTIATE_TESTS=1)

if(ARM OR AARCH64)
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            PLATFORM_ARM)
    if(AARCH64)
        target_compile_definitions(${TARGET_NAME}
            PRIVATE
                PLATFORM_AARCH64)
    endif()
else()
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            PLATFORM_X86)
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/new
        ${CMAKE_CURRENT_SOURCE_DIR}/base
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/helpers
        ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${KMB_TEST_LIBRARIES})

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS VPU KMB)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} EXCLUDE_PATTERNS ".*/new/.*" ALL)
endif()
