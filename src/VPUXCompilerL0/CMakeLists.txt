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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(compilerTest ${CMAKE_CURRENT_SOURCE_DIR}/test/compilerTest.c)
add_executable(profilingTest ${CMAKE_CURRENT_SOURCE_DIR}/test/profilingTest.c)
add_executable(compilerThreadTest ${CMAKE_CURRENT_SOURCE_DIR}/test/compilerThreadTest.cpp)
add_executable(compilerThreadTest2 ${CMAKE_CURRENT_SOURCE_DIR}/test/compilerThreadTest2.cpp)

target_link_libraries(compilerTest
  PUBLIC
  VPUXCompilerL0
)

target_link_libraries(profilingTest
  PUBLIC
  VPUXCompilerL0
)

if(LINUX)
target_link_libraries(compilerThreadTest
  PUBLIC
  VPUXCompilerL0
  pthread
)

target_link_libraries(compilerThreadTest2
  PUBLIC
  VPUXCompilerL0
  pthread
)
else()
target_link_libraries(compilerThreadTest
  PUBLIC
  VPUXCompilerL0
)

target_link_libraries(compilerThreadTest2
  PUBLIC
  VPUXCompilerL0
)
endif()

ie_add_api_validator_post_build_step(TARGET compilerTest)
ie_add_api_validator_post_build_step(TARGET profilingTest)
ie_add_api_validator_post_build_step(TARGET compilerThreadTest)
ie_add_api_validator_post_build_step(TARGET compilerThreadTest2)

set(CID_COMPONENT "CiD")
ie_cpack_add_component(${CID_COMPONENT})

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

    install(FILES $<TARGET_PDB_FILE:compilerThreadTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:compilerThreadTest2>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:compilerTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:profilingTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
else()
    install(TARGETS VPUXCompilerL0
        LIBRARY DESTINATION cid/lib COMPONENT ${CID_COMPONENT})
endif()

install(TARGETS compilerThreadTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(TARGETS compilerThreadTest2
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(TARGETS compilerTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(TARGETS profilingTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})
