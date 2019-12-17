# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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

if(ENABLE_VPUAL AND ENABLE_VPUAL_MODEL)
    list(APPEND VPU_TEST_DEPENDENCIES vpualModel)
endif()

function(addVpuTest TARGET_NAME PLUGIN_NAME PLUGIN_TEST_FOLDER PLUGIN_MACRO_NAME)
    file(GLOB PLUGIN_TEST_INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_TEST_FOLDER}/*.h
            ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_TEST_FOLDER}/*.hpp)
    file(GLOB PLUGIN_TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_TEST_FOLDER}/*.cpp)

    if (PLUGIN_MACRO_NAME STREQUAL "USE_KMB")
        source_group("include" FILES ${TEST_INCLUDE} ${KMB_TEST_INCLUDES} ${PLUGIN_TEST_INCLUDES})
        source_group("src" FILES ${TEST_SRC} ${KMB_TEST_SOURCES} ${PLUGIN_TEST_SOURCES})

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

    add_dependencies(${TARGET_NAME}
            ${DEPENDENCIES}
            ${VPU_TEST_DEPENDENCIES}
            ${PLUGIN_NAME})

    target_compile_options(${TARGET_NAME}
            PRIVATE
            -Wall
            -Werror)

    target_compile_definitions(${TARGET_NAME} PRIVATE
            ${PLUGIN_MACRO_NAME}=ON
            INSTANTIATE_TESTS=1)

    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/base
            ${CMAKE_CURRENT_SOURCE_DIR}/common
            ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/helpers
            ${IE_MAIN_KMB_PLUGIN_SOURCE_DIR}/include
            ${IE_MAIN_SOURCE_DIR}/include/vpu
            ${IE_MAIN_SOURCE_DIR}/src/inference_engine
            ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include)

    target_link_libraries(${TARGET_NAME} PRIVATE
            ${LIBRARIES}
            ${VPU_TEST_LIBRARIES}
            vpusmm
            kmb_utils
        PRIVATE
            IE::vpu_common_lib)

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    set_property(TEST ${TARGET_NAME} PROPERTY LABELS VPU ${PLUGIN_NAME})

    if(ENABLE_KMB_CLANG_FORMAT)
        add_clang_format_target_kmb(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME})
    endif()
endfunction()

addVpuTest(KmbFunctionalTests kmbPlugin kmb_tests USE_KMB)

if (ENABLE_HDDL2)
    add_subdirectory(hddl2_tests)
endif()
