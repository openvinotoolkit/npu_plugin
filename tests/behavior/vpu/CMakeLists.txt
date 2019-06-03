# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

#set(TARGET_NAME ClDnnBehaviorTests)

function(enable_vpu TARGET_NAME FLAG_NAME PLUGIN_NAME)
    file(GLOB TEST_INCLUDE
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests/*.hpp)

    file(GLOB TEST_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/inference_engine_sample_tests/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/cpp_wrappers/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests/*.cpp
            )

    if (ENABLE_MYRIAD)
        file(GLOB VPU_TESTS vpu_tests/*.cpp)
        if (NOT ENABLE_MYRIAD_NO_BOOT)
            list(REMOVE_ITEM VPU_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/vpu_tests/vpu_boot_tests.cpp)
        endif()
        list(APPEND TEST_SRC ${VPU_TESTS})
    endif()


    list(APPEND DEPENDENCIES
            ${PLUGIN_NAME}
            vpu_copy_firmware)

    source_group("src" FILES ${TEST_SRC})
    source_group("include" FILES ${TEST_INCLUDE})

    add_executable(${TARGET_NAME}
            ${TEST_SRC}
            ${REGRESSION_TESTS}
            ${TEST_INCLUDE})

    target_compile_definitions(${TARGET_NAME} PRIVATE
            INSTANTIATE_TESTS=1
            ${FLAG_NAME}=1)

    target_link_libraries(${TARGET_NAME} PRIVATE
    IE::IEBehaviorSharedTests)

    target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests)

    if (ENABLE_MYRIAD)
        target_include_directories(${TARGET_NAME} PRIVATE
                ${CMAKE_SOURCE_DIR}/thirdparty/movidius
                ${CMAKE_SOURCE_DIR}/thirdparty/movidius/mvnc/include
                ${CMAKE_SOURCE_DIR}/thirdparty/movidius/XLink/shared
                ${CMAKE_SOURCE_DIR}/thirdparty/movidius/WinPthread
                ${CMAKE_SOURCE_DIR}/thirdparty/movidius/watchdog)
        target_link_libraries(${TARGET_NAME} PRIVATE mvnc)
    endif()

    add_test(NAME ${TARGET_NAME}
            COMMAND ${TARGET_NAME})

    add_dependencies(${TARGET_NAME} ${DEPENDENCIES})
endfunction(enable_vpu)

if (ENABLE_MYRIAD)
    set(MYRIAD_TARGET_NAME MyriadBehaviorTests)
    enable_vpu(${MYRIAD_TARGET_NAME} USE_MYRIAD myriadPlugin)
endif()

if (ENABLE_HDDL)
    enable_vpu(HddlBehaviorTests USE_HDDL HDDLPlugin)
endif()


    enable_vpu(KmbBehaviorTests USE_KMB kmbPlugin)
