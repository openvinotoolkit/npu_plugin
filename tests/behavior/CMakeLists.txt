# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

function(enable_vpu TARGET_NAME FLAG_NAME PLUGIN_NAME)
    file(GLOB TEST_INCLUDE
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests/*.hpp)

    file(GLOB TEST_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/cpp_wrappers/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests/*.cpp
            )

    source_group("src" FILES ${TEST_SRC})
    source_group("include" FILES ${TEST_INCLUDE})

    add_executable(${TARGET_NAME}
            ${TEST_SRC}
            ${REGRESSION_TESTS}
            ${TEST_INCLUDE})

    target_compile_options(${TARGET_NAME}
            PRIVATE
            -Wall
            -Werror)

    target_compile_definitions(${TARGET_NAME} PRIVATE
            INSTANTIATE_TESTS=1
            ${FLAG_NAME}=1)

    target_link_libraries(${TARGET_NAME}
        PRIVATE
      IE::IEBehaviorSharedTests IE::helpers)

    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests
        SYSTEM PRIVATE
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include)

    add_test(NAME ${TARGET_NAME}
            COMMAND ${TARGET_NAME})
endfunction(enable_vpu)

enable_vpu(KmbBehaviorTests USE_KMB kmbPlugin)
