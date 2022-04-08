#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

set(TARGET_NAME VPUXCompilerL0)
add_library(${TARGET_NAME} SHARED vpux_compiler_l0.cpp)

target_include_directories(${TARGET_NAME}
  PRIVATE
  ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include
  ${IE_MAIN_VPUX_PLUFIN_SOURCE_DIR}/src/vpux_al/include
)

target_link_libraries(${TARGET_NAME}
  PRIVATE
  vpux_al
  INTERFACE IE::inference_engine
)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
