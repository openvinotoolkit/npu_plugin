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
