#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

# vpuxCompilerL0Test as test suit
set(FUNCTIONAL_TARGET vpuxCompilerL0Test)
set(FUNCTIONAL_SOURCES
    vcl_tests_common.cpp
    vcl_tests_single_thread.cpp
    vcl_tests_multiple_compiler.cpp
    vcl_tests_parallel_compilation.cpp)
add_executable(${FUNCTIONAL_TARGET} ${FUNCTIONAL_SOURCES})

if(ENABLE_BLOB_DUMP)
    target_compile_definitions(${FUNCTIONAL_TARGET} PUBLIC BLOB_DUMP=1)
endif()

if(WIN32)
    # Shlwapi.lib is to check file existence.
    target_link_libraries(${FUNCTIONAL_TARGET}
        PUBLIC
            vpux_driver_compiler
            gtest
            gtest_main
            Shlwapi.lib
    )
else()
    target_link_libraries(${FUNCTIONAL_TARGET}
        PUBLIC
            vpux_driver_compiler
            gtest gtest_main pthread
    )
endif()

target_include_directories(${FUNCTIONAL_TARGET}
    PUBLIC
        "${CMAKE_SOURCE_DIR}/src/core/include"
)

if(TARGET IE::nlohmann_json)
    target_include_directories(${FUNCTIONAL_TARGET}
        PRIVATE
            $<TARGET_PROPERTY:IE::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
else()
    find_package(nlohmann_json 3.9.0 QUIET)
    if(nlohmann_json_FOUND)
        target_include_directories(${FUNCTIONAL_TARGET}
            PRIVATE
                $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
    else()
        message(FATAL_ERROR "Failed to find system nlohmann_json.")
    endif()
endif()

link_system_libraries(${FUNCTIONAL_TARGET} PUBLIC IE::ngraph)

ie_add_api_validator_post_build_step(TARGET vpuxCompilerL0Test)

install(TARGETS vpuxCompilerL0Test
        RUNTIME DESTINATION cid
        COMPONENT ${CID_COMPONENT})

if(WIN32)
    install(
            FILES $<TARGET_PDB_FILE:vpuxCompilerL0Test>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
endif()
