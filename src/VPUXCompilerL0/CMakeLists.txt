#
# Copyright Intel Corporation.
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
set(CID_COMPONENT "CiD")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

set(FUNCTIONAL_TARGET vpuxCompilerL0Test)
set(FUNCTIONAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_tests_common.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_tests_thread.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_tests_thread2.cpp)
add_executable(${FUNCTIONAL_TARGET} ${FUNCTIONAL_SOURCES})

add_executable(compilerTest ${CMAKE_CURRENT_SOURCE_DIR}/test/compilerTest.c)
add_executable(profilingTest ${CMAKE_CURRENT_SOURCE_DIR}/test/profilingTest.c)

target_link_libraries(compilerTest
  PUBLIC
  VPUXCompilerL0
)

target_link_libraries(profilingTest
  PUBLIC
  VPUXCompilerL0
)

if(ENABLE_BLOB_DUMP)
    target_compile_definitions(${FUNCTIONAL_TARGET}
    PUBLIC
        BLOB_DUMP=1)
endif()

target_link_libraries(${FUNCTIONAL_TARGET}
        PUBLIC
            VPUXCompilerL0
            gtest
            gtest_main
)

target_include_directories(${FUNCTIONAL_TARGET}
    PUBLIC
        "${CMAKE_SOURCE_DIR}/src/core/include"
)

link_system_libraries(${FUNCTIONAL_TARGET}
    PUBLIC
        IE::ngraph
)

if(LINUX)
  target_link_libraries(${FUNCTIONAL_TARGET}
        PUBLIC
            pthread
    )
endif()

ie_add_api_validator_post_build_step(TARGET compilerTest)
ie_add_api_validator_post_build_step(TARGET profilingTest)
ie_add_api_validator_post_build_step(TARGET vpuxCompilerL0Test)

ie_cpack_add_component(${CID_COMPONENT})

install(TARGETS vpuxCompilerL0Test
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(FILES "${InferenceEngineVPUXPlugin_SOURCE_DIR}/src/VPUXCompilerL0/include/VPUXCompilerL0.h"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})

install(FILES "${InferenceEngineVPUXPlugin_SOURCE_DIR}/src/VPUXCompilerL0/CHANGES.txt"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})

install(FILES "${InferenceEngineVPUXPlugin_SOURCE_DIR}/src/VPUXCompilerL0/README.md"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})

if(WIN32)
    install(TARGETS VPUXCompilerL0
        RUNTIME DESTINATION cid/lib COMPONENT ${CID_COMPONENT})
    install(FILES $<TARGET_PDB_FILE:VPUXCompilerL0>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:compilerTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:profilingTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:vpuxCompilerL0Test>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
else()
    install(TARGETS VPUXCompilerL0
        LIBRARY DESTINATION cid/lib COMPONENT ${CID_COMPONENT})
endif()

install(TARGETS compilerTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(TARGETS profilingTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})
