# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_MCM_COMPILER)
    MESSAGE(STATUS "*** Enabling MCM compiler for functional tests ***")
    add_definitions(-DENABLE_MCM_COMPILER)
else()
    MESSAGE(STATUS "*** Disabling MCM compiler for functional tests ***")
endif()

if (ENABLE_VPUAL)
    add_definitions(-DENABLE_VPUAL)
endif()

file(GLOB VPU_TEST_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/helpers/*.hpp)

file(GLOB VPU_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/helpers/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/common_single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/input_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/io_blob_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/ie_class/*.cpp)

file(GLOB KMB_TEST_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.hpp)
file(GLOB KMB_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)

list(APPEND VPU_TEST_LIBRARIES
    IE::IESharedTests)

function(addVpuTest TARGET_NAME PLUGIN_NAME PLUGIN_TEST_FOLDER PLUGIN_MACRO_NAME)
    file(GLOB PLUGIN_TEST_INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_TEST_FOLDER}/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_TEST_FOLDER}/*.hpp)
    file(GLOB PLUGIN_TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_TEST_FOLDER}/*.cpp)

    if (PLUGIN_MACRO_NAME STREQUAL "USE_KMB")
        source_group("include" FILES ${TEST_INCLUDE} ${KMB_TEST_INCLUDES} ${PLUGIN_TEST_INCLUDES})
        source_group("src" FILES ${TEST_SRC} ${KMB_TEST_SOURCES} ${PLUGIN_TEST_SOURCES})

        message(STATUS "PLUGIN_MACRO_NAME=${PLUGIN_MACRO_NAME}")
        message(STATUS "${KMB_TEST_INCLUDES}")
        message(STATUS "${KMB_TEST_SOURCES}")
        add_executable(${TARGET_NAME}
                ${TEST_INCLUDE} ${TEST_SRC} ${REGRESSION_TESTS}
                ${KMB_TEST_INCLUDES} ${KMB_TEST_SOURCES}
                ${PLUGIN_TEST_INCLUDES} ${PLUGIN_TEST_SOURCES})
    else()
        source_group("include" FILES ${TEST_INCLUDE} ${VPU_TEST_INCLUDES} ${PLUGIN_TEST_INCLUDES})
        source_group("src" FILES ${TEST_SRC} ${VPU_TEST_SOURCES} ${PLUGIN_TEST_SOURCES})
        add_executable(${TARGET_NAME}
                ${TEST_INCLUDE} ${TEST_SRC} ${REGRESSION_TESTS}
                ${VPU_TEST_INCLUDES} ${VPU_TEST_SOURCES}
                ${PLUGIN_TEST_INCLUDES} ${PLUGIN_TEST_SOURCES})
    endif()

    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    add_dependencies(${TARGET_NAME}
            ${DEPENDENCIES}
            ${VPU_TEST_DEPENDENCIES}
            ${PLUGIN_NAME})

    target_compile_definitions(${TARGET_NAME} PRIVATE
            ${PLUGIN_MACRO_NAME}=ON
            INSTANTIATE_TESTS=1)

    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/base
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/helpers
            ${IE_MAIN_SOURCE_DIR}/include/vpu
            ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include
        SYSTEM PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src/kmb_plugin )

    target_link_libraries(${TARGET_NAME} PRIVATE
            ${LIBRARIES}
            ${VPU_TEST_LIBRARIES}
            kmbPlugin_test_static)

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    set_property(TEST ${TARGET_NAME} PROPERTY LABELS VPU ${PLUGIN_NAME})
endfunction()


addVpuTest(KmbFunctionalTests kmbPlugin kmb_tests USE_KMB)
