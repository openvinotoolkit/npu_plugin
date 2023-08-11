#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
set(CID_COMPONENT "CiD")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(ELF_DIR "${InferenceEngineVPUXPlugin_SOURCE_DIR}/thirdparty/graphFile-schema/vpux_elf/")
set(LOADER_DIR_INC "${InferenceEngineVPUXPlugin_SOURCE_DIR}/thirdparty/graphFile-schema/vpux_elf/loader/include/")
set(LOADER_DIR_HPI_INC "${InferenceEngineVPUXPlugin_SOURCE_DIR}/thirdparty/graphFile-schema/vpux_elf/hpi_component/include/")
set(LOADER_DIR_HPI_SRC "${InferenceEngineVPUXPlugin_SOURCE_DIR}/thirdparty/graphFile-schema/vpux_elf/hpi_component/src")
set(LOADER_DIR_SRC "${InferenceEngineVPUXPlugin_SOURCE_DIR}/thirdparty/graphFile-schema/vpux_elf/loader/src")
set(LOADER_DIR "${InferenceEngineVPUXPlugin_SOURCE_DIR}/thirdparty/graphFile-schema/vpux_elf/loader/")

add_subdirectory(src)
add_subdirectory(loader)

set(FUNCTIONAL_TARGET vpuxCompilerL0Test)
set(FUNCTIONAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_tests_common.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_tests_thread.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/vpux_compiler_l0_tests_thread2.cpp)
add_executable(${FUNCTIONAL_TARGET} ${FUNCTIONAL_SOURCES})

add_executable(compilerTest ${CMAKE_CURRENT_SOURCE_DIR}/test/compilerTest.c)
add_executable(profilingTest ${CMAKE_CURRENT_SOURCE_DIR}/test/profilingTest.c)
add_executable(loaderTest ${CMAKE_CURRENT_SOURCE_DIR}/test/loaderTest.cpp)

target_link_libraries(compilerTest
  PUBLIC
        VPUXCompilerL0
)

add_dependencies(VPUXCompilerL0 VPUXLoader)

target_link_libraries(profilingTest
  PUBLIC
        VPUXCompilerL0
)

if(ENABLE_BLOB_DUMP)
    target_compile_definitions(${FUNCTIONAL_TARGET}
    PUBLIC
        BLOB_DUMP=1)
endif()

target_link_libraries(loaderTest
  PUBLIC
        vpux_elf
        vpux_utils
)

# Shlwapi.lib is to check file existence.
if(WIN32)
 target_link_libraries(${FUNCTIONAL_TARGET}
         PUBLIC
             VPUXCompilerL0
             gtest
             gtest_main
             Shlwapi.lib
 )
else()
 target_link_libraries(${FUNCTIONAL_TARGET}
         PUBLIC
             VPUXCompilerL0
             gtest
             gtest_main
             pthread
 )
endif()

target_include_directories(${FUNCTIONAL_TARGET}
    PUBLIC
        "${CMAKE_SOURCE_DIR}/src/core/include"
    PRIVATE
        $<TARGET_PROPERTY:IE::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
)

link_system_libraries(${FUNCTIONAL_TARGET}
    PUBLIC
        IE::ngraph
)

ie_add_api_validator_post_build_step(TARGET compilerTest)
ie_add_api_validator_post_build_step(TARGET profilingTest)
ie_add_api_validator_post_build_step(TARGET vpuxCompilerL0Test)
ie_add_api_validator_post_build_step(TARGET loaderTest)

ov_cpack_add_component(${CID_COMPONENT})

install(TARGETS vpuxCompilerL0Test
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(FILES "${InferenceEngineVPUXPlugin_SOURCE_DIR}/src/VPUXCompilerL0/include/VPUXCompilerL0.h"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})

#--------------------------------------------------------------------------------------------
# vpux_elf lib + loader lib
#--------------------------------------------------------------------------------------------
#
# loader folder | -> include/ | -> vpux_headers/ -> 1 hpp
#               |             | -> vpux_loader/  -> 1 hpp

install(DIRECTORY "${LOADER_DIR_INC}/vpux_headers"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_INC}/vpux_loader"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

#
# hpi_component | -> common/ -> 2 hpp
#               | -> 3720/   -> 1 hpp

install(DIRECTORY "${LOADER_DIR_HPI_INC}/common"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_HPI_INC}/3720"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

#
# loader folder | -> src/ | 1 cpp

install(DIRECTORY "${LOADER_DIR_SRC}/"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})

#
# hpi_component | src | common -> 1 cpp
#               |     |  3720  -> 1 cpp

install(DIRECTORY "${LOADER_DIR_HPI_SRC}/common"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_HPI_SRC}/3720"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})

# core folder | -> include/ -> vpux_elf/ | -> types/
#             |                          | -> utils/
#             |                          | -> writer/
#             |                          | -> + 3 hpp

# 3 hpp
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf"
        DESTINATION cid/vpux_elf/core/include/
        COMPONENT ${CID_COMPONENT})

# types
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf/types"
        DESTINATION cid/vpux_elf/core/include/vpux_elf/
        COMPONENT ${CID_COMPONENT})

# utils
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf/utils"
        DESTINATION cid/vpux_elf/core/include/vpux_elf/
        COMPONENT ${CID_COMPONENT})

# writer
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf/writer"
        DESTINATION cid/vpux_elf/core/include/vpux_elf/
        COMPONENT ${CID_COMPONENT})

# core folder | -> src/ -> | -> types/
#             |            | -> utils/
#             |            | -> writer/
#             |            | -> + 2 cpp

# 2 cpp
install(DIRECTORY "${ELF_DIR}/core/src"
        DESTINATION cid/vpux_elf/core/
        COMPONENT ${CID_COMPONENT})

# types
install(DIRECTORY "${ELF_DIR}/core/src/types"
        DESTINATION cid/vpux_elf/core/src/
        COMPONENT ${CID_COMPONENT})

# utils
install(DIRECTORY "${ELF_DIR}/core/src/utils"
        DESTINATION cid/vpux_elf/core/src/
        COMPONENT ${CID_COMPONENT})

# writer
install(DIRECTORY "${ELF_DIR}/core/src/writer"
        DESTINATION cid/vpux_elf/core/src/
        COMPONENT ${CID_COMPONENT})

#--------------------------------------------------------------------------------------------

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

    install(FILES $<TARGET_PDB_FILE:loaderTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
else()
    install(TARGETS VPUXCompilerL0
        LIBRARY DESTINATION cid/lib COMPONENT ${CID_COMPONENT})

    install(TARGETS VPUXLoader
        LIBRARY DESTINATION cid/lib
        COMPONENT ${CID_COMPONENT})
endif()

install(TARGETS VPUXLoader
        CONFIGURATIONS Release
        LIBRARY DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT})
install(TARGETS VPUXLoader
        CONFIGURATIONS Debug
        LIBRARY DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT})
install(TARGETS VPUXLoader
        CONFIGURATIONS RelWithDebInfo
        LIBRARY DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT})

install(TARGETS vpux_elf
        CONFIGURATIONS Release
        LIBRARY DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT})
install(TARGETS vpux_elf
        CONFIGURATIONS Debug
        LIBRARY DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT})
install(TARGETS vpux_elf
        CONFIGURATIONS RelWithDebInfo
        LIBRARY DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT})

target_compile_definitions(vpux_elf PUBLIC HOST_BUILD)

install(TARGETS compilerTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(TARGETS profilingTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})

install(TARGETS loaderTest
        RUNTIME DESTINATION cid COMPONENT ${CID_COMPONENT})
