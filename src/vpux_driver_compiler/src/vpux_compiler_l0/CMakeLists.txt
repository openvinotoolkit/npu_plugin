#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME vpux_driver_compiler)

add_library(${TARGET_NAME}
    SHARED
        vcl_bridge.cpp
        vcl_common.cpp
        vcl_compiler.cpp
        vcl_executable.cpp
        vcl_profiling.cpp
        vcl_query_network.cpp
)

# Use commit info to create compiler version
set(repo_root ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR})
commitHash(PLUGIN_GIT_COMMIT_HASH)
string(TIMESTAMP CURRENT_DATE "%Y-%m-%d-%H-%M-%S")
set(DRIVER_COMPILER_ID ${CI_BUILD_NUMBER}-${CURRENT_DATE}-${OpenVINO_VERSION_BUILD}-DCI-${PLUGIN_GIT_COMMIT_HASH})
target_compile_definitions(${TARGET_NAME}
  PUBLIC
  DRIVER_COMPILER_ID=${DRIVER_COMPILER_ID}
)

# Add version to library
set(OV_VS_VER_FILEVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
set(OV_VS_VER_PRODUCTVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
set(OV_VS_VER_FILEVERSION_STR "${OV_VS_VER_FILEVERSION_QUAD}")
set(OV_VS_VER_PRODUCTVERSION_STR "${DRIVER_COMPILER_ID}")
set(OV_VS_VER_COMPANY_NAME_STR "Intel Corporation")
set(OV_VS_VER_PRODUCTNAME_STR "VPU Compiler In Driver")
set(OV_VS_VER_COPYRIGHT_STR "Copyright (C) 2018-2023, Intel Corporation")
set(OV_VS_VER_FILEDESCRIPTION_STR "VPU Plugin Compiler L0")
set(OV_VS_VER_ORIGINALFILENAME_STR "${CMAKE_SHARED_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(OV_VS_VER_INTERNALNAME_STR ${TARGET_NAME})
set(vs_version_output "${CMAKE_CURRENT_BINARY_DIR}/vs_version.rc")
configure_file("${IEDevScripts_DIR}/vs_version/vs_version.rc.in" "${vs_version_output}" @ONLY)
source_group("src" FILES ${vs_version_output})
target_sources(${TARGET_NAME} PRIVATE ${vs_version_output})

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUFIN_SOURCE_DIR}/src/vpux_al/include)

target_link_libraries(${TARGET_NAME} PRIVATE npu_al)

add_dependencies(${TARGET_NAME} VPUXLoader)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

if(WIN32)
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION cid/lib
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:vpux_driver_compiler>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
else()
    install(TARGETS ${TARGET_NAME}
            LIBRARY DESTINATION cid/lib
            COMPONENT ${CID_COMPONENT})
endif()

install(
        FILES
            "${InferenceEngineVPUXPlugin_SOURCE_DIR}/src/vpux_driver_compiler/include/VPUXCompilerL0.h"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})

# Duplicate vpux_driver_compiler for backward compatibility

set(OLD_TARGET_NAME VPUXCompilerL0)
# Build old target before new target
add_dependencies(${TARGET_NAME} ${OLD_TARGET_NAME})

add_library(${OLD_TARGET_NAME}
    SHARED
        vcl_bridge.cpp
        vcl_common.cpp
        vcl_compiler.cpp
        vcl_executable.cpp
        vcl_profiling.cpp
        vcl_query_network.cpp
)

target_compile_definitions(${OLD_TARGET_NAME}
  PUBLIC
      DRIVER_COMPILER_ID=${DRIVER_COMPILER_ID}
)

target_sources(${OLD_TARGET_NAME} PRIVATE ${vs_version_output})

target_include_directories(${OLD_TARGET_NAME}
    PRIVATE
        ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
        ${IE_MAIN_VPUX_PLUFIN_SOURCE_DIR}/src/vpux_al/include
)

target_link_libraries(${OLD_TARGET_NAME} PRIVATE npu_al)

add_dependencies(${OLD_TARGET_NAME} VPUXLoader)

ie_add_api_validator_post_build_step(TARGET ${OLD_TARGET_NAME})

if(WIN32)
    install(TARGETS ${OLD_TARGET_NAME}
            RUNTIME DESTINATION cid/lib
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:VPUXCompilerL0>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
else()
    install(TARGETS ${OLD_TARGET_NAME}
            LIBRARY DESTINATION cid/lib
            COMPONENT ${CID_COMPONENT})
endif()

install(
        FILES
            "${InferenceEngineVPUXPlugin_SOURCE_DIR}/src/vpux_driver_compiler/include/vpux_driver_compiler.h"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})
