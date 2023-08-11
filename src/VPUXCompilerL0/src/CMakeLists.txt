#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME VPUXCompilerL0)
add_library(${TARGET_NAME} SHARED vpux_compiler_l0.cpp)

# Use commit info to create compiler version
set(repo_root ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR})
commitHash(PLUGIN_GIT_COMMIT_HASH)
set(DRIVER_COMPILER_ID ${CI_BUILD_NUMBER}-DCI-${PLUGIN_GIT_COMMIT_HASH})
target_compile_definitions(${TARGET_NAME}
  PUBLIC
  DRIVER_COMPILER_ID=${DRIVER_COMPILER_ID}
)

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
  ${IE_MAIN_VPUX_PLUFIN_SOURCE_DIR}/src/vpux_al/include
)

target_link_libraries(${TARGET_NAME}
  PRIVATE
  vpux_al
)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
