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

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(TARGET_NAME vpux_al)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

enable_warnings_as_errors(${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/include")

set(VPUX_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp)
addVersionDefines(${VPUX_VERSION_FILE} CI_BUILD_NUMBER)

link_system_libraries(${TARGET_NAME}
    PRIVATE
        IE::inference_engine
        IE::vpu_common_lib)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        kmb_utils
    PUBLIC
        vpux_utils)

if(ENABLE_CLANG_FORMAT)
    add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME} ALL)
    add_dependencies(${TARGET_NAME} clang_format_${TARGET_NAME})
endif()
